"""Protected immutable local records and deterministic SQLite outbox recovery."""

import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

from anyio import to_thread
from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from emergence.telemetry.logging import SECRET_FIELD, Redactor

from .capacity import live_claims
from .database import Database
from .models import archive_outbox, archive_references
from .repositories import (
    ArchiveReference,
    list_archive_references,
    register_archive_reference,
)

_RECORD: Final = re.compile(r"[a-f0-9]{64}")
_STAGING: Final = re.compile(r"\.stage-[a-zA-Z0-9_-]+")
_FORMAT: Final = b"emergence-spool-v1"


class SpoolError(RuntimeError):
    """A protected spool invariant failed; recovery must fail closed."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RecordKey:
    """Opaque stable identity, record kind and zero-based ordinal."""

    source_id: str
    kind: str
    ordinal: int = 0

    @property
    def record_id(self) -> str:
        identity = json.dumps([self.source_id, self.kind, self.ordinal], ensure_ascii=True)
        return sha256(_FORMAT + b"\0" + identity.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class _Record:
    reference: ArchiveReference
    data: bytes


@dataclass(frozen=True, slots=True)
class SpoolStats:
    used_bytes: int
    records: int
    free_bytes: int


def _scrub(redactor: Redactor, payload: JsonValue) -> JsonValue:
    # Also protect JSON property names, which the logging redactor preserves.
    match payload:
        case dict():
            output: dict[str, JsonValue] = {}
            for key, value in payload.items():
                safe_key = redactor.text(key)
                if safe_key in output:
                    raise SpoolError("redacted_key_collision")
                output[safe_key] = (
                    "[REDACTED]" if SECRET_FIELD.search(key) else (_scrub(redactor, value))
                )
            return output
        case list():
            return [_scrub(redactor, value) for value in payload]
        case _:
            return redactor.value(payload)


def _stamp(record_id: str, body: bytes) -> bytes:
    return sha256(_FORMAT + b"\0" + record_id.encode("ascii") + b"\0" + body).hexdigest().encode()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Spool:
    """Single local spool per database; pass a redactor configured with known secrets."""

    def __init__(
        self,
        path: Path,
        database: Database,
        redactor: Redactor,
        *,
        limit_bytes: int = 256 * 1024 * 1024,
        headroom_bytes: int = 64 * 1024 * 1024,
        archive_path: Path | None = None,
        archive_mount_path: Path | None = None,
    ) -> None:
        from .delivery import MountedArchive

        self._path = path.absolute()
        self._db = database
        self._redactor = redactor
        self._limit = limit_bytes
        self._headroom = headroom_bytes
        self.archive = (
            MountedArchive(archive_path, archive_mount_path or archive_path, self._path)
            if archive_path is not None
            else None
        )

    def preview(self, key: RecordKey, payload: JsonValue) -> ArchiveReference:
        """Compute exact redacted input storage before financial admission."""
        return self._prepare(key, payload).reference

    async def stats(self) -> SpoolStats:
        return await to_thread.run_sync(self._stats)

    async def readiness(self) -> bool:
        """Check the local spool without creating it or contacting the archive."""
        try:
            return await to_thread.run_sync(self._directory_usable)
        except (OSError, SpoolError):
            return False

    def _directory_usable(self) -> bool:
        self._check_directory()
        return os.access(self._path, os.R_OK | os.W_OK | os.X_OK, effective_ids=True)

    async def ensure_capacity(
        self,
        session: AsyncSession,
        additional_bytes: int,
        *,
        claim_request_id: str | None = None,
        record_id: str | None = None,
        record_size: int = 0,
    ) -> None:
        """Check bytes and disk headroom under the caller's SQLite write lock."""
        claims = await live_claims(session)
        if any(claim.remaining_bytes < 0 for claim in claims):
            raise SpoolError("spool_claim_exceeded")
        reduction = 0
        if claim_request_id is not None:
            own = next((claim for claim in claims if claim.request_id == claim_request_id), None)
            if own is None or record_id not in (own.input_record_id, own.output_record_id):
                raise SpoolError("spool_claim_missing")
            if record_size > own.remaining_bytes:
                raise SpoolError("spool_claim_exceeded")
            reduction = record_size
        remaining = sum(claim.remaining_bytes for claim in claims) - reduction
        stats = await self.stats()
        if stats.used_bytes + remaining + additional_bytes > self._limit:
            raise SpoolError("spool_capacity_exhausted")
        if stats.free_bytes < self._headroom + remaining + additional_bytes:
            raise SpoolError("local_disk_headroom_exhausted")

    async def publish(
        self, key: RecordKey, payload: JsonValue, *, claim_request_id: str | None = None
    ) -> ArchiveReference:
        """Redact and durably publish an immutable JSON record, then commit its reference."""
        record = self._prepare(key, payload)
        reference = record.reference
        async with self._db.write() as session:
            existing = (
                await session.execute(
                    select(archive_references).where(
                        archive_references.c.record_id == reference.record_id
                    )
                )
            ).one_or_none()
            if existing is not None:
                if (existing.checksum, existing.size_bytes) != (
                    reference.checksum,
                    reference.size_bytes,
                ):
                    raise SpoolError("record_identity_conflict")
                try:
                    await self.read_local(reference)
                except SpoolError as error:
                    # A regenerated identical record (notably startup configuration)
                    # already has a durable acknowledgement. Never consult NAS here.
                    if error.code != "missing_record" or not await _archived(session, reference):
                        raise
            else:
                present = await to_thread.run_sync((self._path / reference.record_id).exists)
                await self.ensure_capacity(
                    session,
                    0 if present else reference.size_bytes,
                    claim_request_id=claim_request_id,
                    record_id=reference.record_id,
                    record_size=reference.size_bytes,
                )
                await to_thread.run_sync(self._publish, record)
            await register_archive_reference(session, reference)
        return reference

    async def read(self, reference: ArchiveReference) -> bytes:
        """Return the verified on-disk bytes covered by the reference's SHA-256."""
        try:
            return await self.read_local(reference)
        except SpoolError as error:
            if error.code != "missing_record" or self.archive is None:
                raise
            async with self._db.session() as session:
                if not await _archived(session, reference):
                    raise
        return await self.archive.read(reference)

    async def read_local(self, reference: ArchiveReference) -> bytes:
        return await to_thread.run_sync(self._read_reference, reference)

    async def local_ids(self) -> set[str]:
        return await to_thread.run_sync(
            lambda: {p.name for p in self._path.iterdir() if _RECORD.fullmatch(p.name)}
        )

    def reclaim(self, reference: ArchiveReference) -> None:
        """Local-only; caller holds the writer after verified delivery and committed ACK."""
        if not _RECORD.fullmatch(reference.record_id):
            raise SpoolError("invalid_record_id")
        self._check_directory()
        (self._path / reference.record_id).unlink()
        _fsync_directory(self._path)

    def _read_reference(self, reference: ArchiveReference) -> bytes:
        self._check_directory()
        record = self._read(reference.record_id)
        if record.reference != reference:
            raise SpoolError("reference_mismatch")
        return record.data

    async def recover(self) -> None:
        """Reconcile durable files and rows; do not silently drop corrupt or missing data."""
        async with self._db.write() as session:
            known = await list_archive_references(session)
            archived = set(
                await session.scalars(
                    select(archive_outbox.c.record_id).where(archive_outbox.c.state == "archived")
                )
            )
            recovered = await to_thread.run_sync(self._scan, known, archived)
            for reference in recovered:
                await register_archive_reference(session, reference)

    def _ensure_directory(self) -> None:
        try:
            # Create only the leaf. Provision its local parent before launching.
            self._path.mkdir(mode=0o700)
        except FileExistsError:
            self._check_directory()
        _fsync_directory(self._path.parent)

    def _check_directory(self) -> None:
        metadata = self._path.lstat()
        if metadata.st_uid != os.geteuid() or metadata.st_mode != stat.S_IFDIR | 0o700:
            raise SpoolError("unsafe_spool_directory")

    def _read(self, record_id: str) -> _Record:
        if _RECORD.fullmatch(record_id) is None:
            raise SpoolError("invalid_record_id")
        try:
            fd = os.open(self._path / record_id, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            raise SpoolError("missing_record") from None
        with os.fdopen(fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if metadata.st_uid != os.geteuid() or metadata.st_mode != stat.S_IFREG | 0o400:
                raise SpoolError("unsafe_record")
            data = stream.read()
        version, _, remainder = data.partition(b"\n")
        checksum, _, content = remainder.partition(b"\n")
        if version != _FORMAT or not content or checksum != _stamp(record_id, content):
            raise SpoolError("record_checksum_mismatch")
        return _Record(ArchiveReference(record_id, sha256(data).hexdigest(), len(data)), data)

    def _prepare(self, key: RecordKey, payload: JsonValue) -> _Record:
        scrubbed = _scrub(self._redactor, payload)
        try:
            body = json.dumps(
                scrubbed, ensure_ascii=True, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode("ascii")
        except ValueError:
            raise SpoolError("invalid_json_payload") from None
        record_id = key.record_id
        data = b"\n".join((_FORMAT, _stamp(record_id, body), body))
        reference = ArchiveReference(record_id, sha256(data).hexdigest(), len(data))
        return _Record(reference, data)

    def _stats(self) -> SpoolStats:
        self._ensure_directory()
        total = 0
        records = 0
        for entry in self._path.iterdir():
            try:
                metadata = entry.lstat()
            except FileNotFoundError:
                # An unlocked status snapshot may race publication or reclamation.
                continue
            if not stat.S_ISREG(metadata.st_mode) or not (
                _RECORD.fullmatch(entry.name) or _STAGING.fullmatch(entry.name)
            ):
                raise SpoolError("unexpected_spool_entry")
            total += metadata.st_size
            records += int(_RECORD.fullmatch(entry.name) is not None)
        return SpoolStats(total, records, shutil.disk_usage(self._path).free)

    def _publish(self, record: _Record) -> None:
        reference, data = record.reference, record.data
        record_id = reference.record_id
        self._ensure_directory()
        if (self._path / record_id).exists():
            if self._read(record_id).reference != reference:
                raise SpoolError("record_identity_conflict")
            _fsync_directory(self._path)
            return
        with NamedTemporaryFile(dir=self._path, prefix=".stage-", delete=False) as staged:
            try:
                staged.write(data)
                staged.flush()
                os.fchmod(staged.fileno(), 0o400)
                os.fsync(staged.fileno())
                try:
                    os.link(staged.name, self._path / record_id)
                except FileExistsError:
                    if self._read(record_id).reference != reference:
                        raise SpoolError("record_identity_conflict") from None
                # A previous interrupted publication may have linked but not synced.
                _fsync_directory(self._path)
            finally:
                Path(staged.name).unlink()
                _fsync_directory(self._path)

    def _scan(
        self, known: tuple[ArchiveReference, ...], archived: set[str]
    ) -> tuple[ArchiveReference, ...]:
        self._ensure_directory()
        discovered: dict[str, ArchiveReference] = {}
        stages: list[Path] = []
        for entry in sorted(self._path.iterdir()):
            if _RECORD.fullmatch(entry.name):
                discovered[entry.name] = self._read(entry.name).reference
            elif _STAGING.fullmatch(entry.name) and stat.S_ISREG(entry.lstat().st_mode):
                stages.append(entry)
            else:
                raise SpoolError("unexpected_spool_entry")
        for reference in known:
            observed = discovered.get(reference.record_id)
            if observed is None:
                if reference.record_id in archived:
                    continue
                raise SpoolError("missing_record")
            if observed != reference:
                raise SpoolError("reference_mismatch")
        for stage in stages:
            stage.unlink()
        _fsync_directory(self._path)
        return tuple(discovered.values())


async def _archived(session: AsyncSession, reference: ArchiveReference) -> bool:
    return (
        await session.scalar(
            select(archive_outbox.c.state)
            .join(archive_references)
            .where(
                archive_references.c.record_id == reference.record_id,
                archive_references.c.checksum == reference.checksum,
                archive_references.c.size_bytes == reference.size_bytes,
            )
        )
    ) == "archived"
