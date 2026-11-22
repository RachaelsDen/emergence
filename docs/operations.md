# Operating the trusted core

## Install and supervise

The charter records an initial unprivileged Debian 13 Proxmox LXC allocation
of 2 vCPU, 2 GiB RAM and 24 GiB local disk. This is observed baseline context,
not a provisioning step or capacity guarantee. `172.17.2.200` is a historical
LAN address, not a listen default. Leave VM 201 and the autonomous-agent
runtime untouched. **Do not expose admin or inference on `10.10.10.0/24`
(the agent bridge)**, whether directly, through a proxy, or through firewall
forwarding. Keep `127.0.0.1` by default. If remote operator access is needed,
explicitly authorize a separate management network and restrict ingress at
the host/Proxmox boundary (IPv4 and IPv6); use authenticated encrypted
transport, e.g. a protected management tunnel or TLS proxy. A bearer token
alone is not a network boundary.

The following is an operator installation procedure, not a claim that the
target unit is installed.

1. Provision a dedicated unprivileged `emergence` system user and group with
   no interactive login. Put a reviewed checkout in `/opt/emergence`. From
   that directory, use Python 3.12+ and run `uv sync --locked --no-dev` as a
   trusted installer, not the runtime user. Keep the code and `.venv`
   readable/executable but not writable by `emergence`. Ensure the venv's
   Python resolves outside protected home directories (e.g. system Python or
   a trusted executable location under `/opt`).
2. Copy `config.example.yaml` to `/etc/emergence/config.yaml`. Use a
   root-owned config readable by the service (e.g. `root:emergence`, `0640`);
   keep `/etc/emergence` root-controlled. Override YAML `database_path` to
   `/var/lib/emergence/core.sqlite3` and `spool_path` to
   `/var/lib/emergence/spool`. Relative paths resolve from the process working
   directory, not the YAML file. Keep SQLite and its sidecars on local disk.
3. Create `/etc/emergence/secrets.env` owned by root with mode `0600`. Use
   systemd `NAME=value` entries with the names in YAML for any configured
   bearer or provider credentials. PID 1 reads the file; the service user
   does not need read permission on it. Do not commit it, put secret values
   on the command line, or export them to guest or agent environments.
4. Before first start (and with the service stopped on upgrade), create the
   local database parent and spool directory owned by `emergence`, mode
   `0700`; back up existing state before migrating. From `/opt/emergence`,
   as an authorized administrator, run:

   ```sh
   sudo install -d -o emergence -g emergence -m 0700 /var/lib/emergence /var/lib/emergence/spool
   sudo -u emergence env -u EMERGENCE_DATABASE_PATH \
     EMERGENCE_CONFIG=/etc/emergence/config.yaml \
     sh -c 'umask 077; exec /opt/emergence/.venv/bin/alembic upgrade head'
   ```

   Alembic alone can override its target using `EMERGENCE_DATABASE_PATH`;
   unset it when migrating the YAML target. Verify the migration and ownership.
   Alembic enables SQLite WAL mode. For an existing database, rerun the same
   command with the service stopped, even if it is already at the current
   revision. Readiness requires WAL mode; startup does not change it or create
   or migrate the database.
5. Install `deploy/systemd/emergence.service` under
   `/etc/systemd/system/`. Run `systemctl daemon-reload`, then
   `systemctl enable --now emergence.service`. Check
   `systemctl status emergence.service`, `journalctl -u emergence.service`,
   and local liveness and readiness. On shutdown or restart use `systemctl
   stop emergence.service` or `systemctl restart emergence.service`; do not
   start a second core writer. Wait for exit before state maintenance; check
   readiness, resource/backlog status and unresolved requests on restart.

The unit runs the venv executable directly with `/opt/emergence` as its
working directory. `StateDirectory=emergence` leaves local state writable
under `ProtectSystem=strict`; outbound DNS/HTTPS and access to the local
inference listener are not disabled by the unit. It deliberately has no NAS
mount dependency. For later archive enablement with a distinct path, add a
root-owned systemd drop-in (example using the required mount root):

```ini
[Service]
ReadWritePaths=-/mnt/emergence-archive
```

Restrict this exception to an operator-controlled archive mount (or a
dedicated existing archive directory on it), not all of `/mnt`. The `-`
prefix makes missing paths nonfatal for systemd; it does not bypass the
application's mount check. Verify mount visibility, write permissions and
mount propagation in the service's namespace, including outage/reappearance.
If a path was absent when systemd set up its namespace, a restart may be
needed to apply the write exception. Sandbox behavior must be verified on the
actual unprivileged LXC.

## Configure provider access and bounds

Enable a provider only after selecting its exact entitled model ID and
checking the endpoint, credentials and applicable limits. Each compatible
`base_url` includes the API prefix; the adapter appends `/chat/completions`.
The examples are candidates, not tested account settings.

- **Z.ai:** match the coding-plan credential to its coding-plan inference
  endpoint; do not substitute a PAYG key or endpoint. Quota is separate from
  inference; the undocumented quota interpretation remains experimental.
- **MiniMax:** verify that the configured compatible base accepts the
  coding-plan key in the intended region; do not silently switch to PAYG.
  Verify its quota response separately. The adapter treats reasoning as
  mandatory even if the caller does not request it. Cost enforcement needs
  evidence that reasoning tokens overlap and are included inside the same
  verified `max_completion_tokens` bound, not charged as an independent
  unbounded amount.
- **OpenAI:** set the intended `organization` and `project`, and put the
  project service-account inference key and separate restricted Admin
  reporting key in their respective named environment variables.
  Reporting uses the organization usage/cost endpoints and scoped project
  data; a working inference key does not prove reporting access. Maintain
  `reporting_key_expires_at` and investigate expiry warnings. Confirm actual
  credit assignment and account spending/auto-reload settings independently.
- **llama.cpp:** set the compatible base and exact model ID for a trusted
  server reachable from the LXC. If it requires a bearer token, set a named
  `credential_env`. Track bounded compute, not a dollar balance.

Z.ai and MiniMax reporting may fall back to their own inference credential.
Do not reuse an inference credential from another configured provider for
reporting, even if that provider is disabled.

The example starts at one concurrent request per provider, a 30-second
deadline, 4096 conservative input units and at most 256 output tokens per
request. Input admission counts UTF-8 bytes plus configured per-message
protocol overhead. Verify that overhead and the upstream input/output
mechanisms for the actual model and endpoint. Do not turn any bound flag
on without matching sanitized evidence in `input_bound_evidence`,
`bound_evidence`, or `reasoning_bound_evidence`. Unverified applicable
billable bounds block cost-enforced dispatch; unsupported tools and
extensions are rejected. Local caps are not proof of provider behavior.

For enabled cloud aliases, supply real versioned prices. Each `prices` item
needs `provider` (configured name), `model` (alias), `version`, a
timezone-aware `effective_at`, `source` (sanitized provenance), and exact
quoted decimal `input_per_million` and `output_per_million` rates in USD per
**million** tokens. Supply `cached_input_per_million` only if applicable and
sourced; unknown cache usage is charged at the input rate. For coding plans,
use the verified equivalent PAYG rates for **simulated** charges, not a
pretended subscription balance. Input and output rates must be positive;
a documented cached-input rate may be zero. Do not use dummy rates.
Append a version when rates change; preserve historical definitions.
Missing effective rates are not an estimate of zero spend.

## Budgets and quota

Each cloud provider has one USD 100 virtual budget shared by its users, not
one per request, model or future agent. Periods are calendar months with
`accounting_timezone: UTC` by default (configurable IANA zone), no rollover,
and preserved history. Choose the timezone before opening a cloud budget period.
Changing it for an enabled provider with an existing ledger is not supported:
readiness fails with `accounting_timezone_changed` in the log. Restore the
stored zone; do not edit historical periods. The ledger reserves before
dispatch and settles with the admitted price version; missing cloud usage can
retain the full hold pending audited investigation. Grants, reductions and corrections affect
virtual accounting, not provider credits. Avoid multiple configured names
for one intended OpenAI account scope: exposure is provider-name keyed.
The opening is fixed at USD 100; the compatibility `monthly_budget` setting
accepts only `100`. Change spendable virtual funds through audited adjustments.
Every adjustment requires a stable operation key (`Idempotency-Key` over HTTP,
`--adjustment-id` in the CLI). Save the generated or supplied key from CLI
output and reuse it with identical inputs when a reply is lost. Replays retain
the original period and cannot grant funds again after renewal. Conflicting
inputs return 409. These identifiers grant no authority without operator
authentication and are not bearer credentials.

Keep measured scoped provider report aggregates, local estimates and virtual
charges separate. Aggregate reports are not measured costs for individual
requests, can lag or revise, and overlapping UTC buckets can appear in more
than one non-UTC period projection. The independent OpenAI USD 95 safety
limit requires `safety_limit_confirmed: true` only after operator confirmation
of the policy and intended scope. Its exposure accounts conservatively for
scoped measured costs vs estimates, uncovered estimates, and open holds;
it does not claim actual remaining credit or guarantee zero overage. A new
virtual month does **not** reset known provider exhaustion or replenish that
independent OpenAI safety exposure.

`local_limits_only` can admit a request with missing or stale quota when
other local bounds permit it; known insufficient provider capacity still
blocks. `require_fresh` also requires applicable fresh capacity evidence
from the latest observation; a newer failed or ambiguous refresh invalidates
an older positive snapshot for this policy. It can deny when the provider
cannot offer fresh capacity. Neither policy turns
unknown observations into zero or unlimited capacity. Check freshness,
confidence, scope and available provenance in `/api/v1/resources`.
Known USD capacity includes local commitments not covered by its snapshot,
including an earlier call still in flight when that snapshot was observed.

An inference-observed exhaustion denial remains blocking across time, restarts,
and virtual-month renewal. Reporting for another resource cannot clear it.
After independently confirming replenishment for the intended provider account,
an operator can use the authenticated
`POST /api/v1/providers/{provider}/exhaustion/resolution` endpoint documented in
[the API contract](api.md), supplying the service-issued request ID of the
current denial and a bounded evidence-based reason. Do not put credentials in
the reason. There is no automatic expiry, provider retry, or inference dispatch
in this operation.

The operation appends a boolean availability assertion and an audit event
atomically after archiving redacted evidence; it preserves the original denial.
A crash or concurrent denial between archive publication and the transaction
can leave an unreferenced recovery detail record, which does not authorize
admission. A duplicate or stale confirmation returns `409`. Inspect the current
resource and request history before deciding whether a new confirmation is
warranted. The API does not confirm any actual credit amount, release ambiguous
holds, reset OpenAI safety exposure, or override another known exhausted
resource. Those independent gates can still deny inference after recovery.
For `require_fresh`, the assertion lasts only `stale_after_seconds`; a stale
assertion remains unknown under that policy and cannot be renewed by replaying
the old denial's resolution. After independently checking availability again,
include the current recovery receipt's `observation_id` in a new confirmation.
The API rejects a stale ID, so renewed evidence cannot clear a newer denial or
refresh authority by replaying an old confirmation.

## Archive, retention and monitoring

Set `archive_path` to a protected NAS-backed directory; set the optional
`archive_mount_path` to the actual mount root if the archive directory is
below it (otherwise the archive path itself must be a mount). Provision
and mount the share externally through the operator/Proxmox, never through
the application. Keep active SQLite and its WAL on local disk, **never
on NAS**. Pinned Linux descriptor mount-ID checks reject an ordinary
unmounted directory and support bind mounts. Require service-user access,
POSIX modes, hard links, file fsync and directory fsync; mount presence is
neither proof of the intended share's identity nor proof of persistence
through server power loss.

During NAS outages local metadata remains ready if local storage is healthy.
Resources retain SQL-backed numeric observations and show
`archive_detail_unavailable` when reclaimed reporting detail cannot be read.
Scope and detail are left unknown rather than reconstructed from current
configuration. Remounting restores verified detail; missing records or checksum
damage still fail explicitly.
Reads that need the unavailable archive can fail; queued payloads stay local
and bounded spool capacity/headroom denies new dispatch under pressure.
NAS work uses one daemon I/O lane shared with remote reads; pending work is
bounded and shutdown does not wait for a stalled NAS syscall. The worker
checks payload size and checksum, syncs the destination file and directory
and commits durable acknowledgement before reclaim. Unresolved requests'
input and output stay local even if copied; reclaim needs both verified NAS
durability with ACK and eligibility. No age-based purge.

Use authenticated `/api/v1/resources` for observations, spool size/pressure,
and archive state even when `metrics_enabled: false`. With metrics enabled,
scrape bearer-protected `/metrics` with a reader token. Request and observed
token totals are derived from durable authority; inference duration
accumulates in-process and resets on restart. Prometheus monetary values are
approximate monitoring gauges; exact decimal ledger/API strings are the
financial authority. Alert on spool pressure, archive backlog/failure,
reporting unavailable/stale, and low budget/near-limit exposure.

## Backup, restore, and exceptional removal

Prefer a stopped-core backup, or use SQLite's consistent backup mechanism
(e.g. the SQLite backup API) with a coordinated payload snapshot or paused
publication/reclaim. A live copy of just the SQLite database file can omit
committed WAL data; do not take one. For a stopped raw-file backup, close
all SQLite users and verify a successful checkpoint before copying the
standalone database, or preserve the required WAL with its matching database
from the same consistent snapshot. Do not combine files from different points
in time.

Capture the matching immutable NAS payloads **and** local spool (including
pending and unresolved data). Record a protected manifest of the DB/revision
and referenced payload IDs, byte sizes, checksums and locations. Keep the
config with the backup and secrets separately under access control. Check
SQLite `PRAGMA integrity_check` and `PRAGMA foreign_key_check`, match
reference/acknowledgement state, and verify envelope sizes and hashes.
Test restore into isolated local paths with a separate archive target and
provider egress blocked. Verify migrations/readiness, ledger continuity,
pending and unresolved state, and representative authenticated history
reads before exposing the restored service. Restoring only SQLite cannot
recreate archived payloads.

Retain histories indefinitely by default, subject to exceptional authorized
privacy, secret-redaction or legal purges. Stop and coordinate with the
operator before such a change. Record who authorized it, scoped record IDs,
reason and time without repeating sensitive payloads. Preserve immutable
ledger/events/references and audit metadata; never rewrite a checksummed
envelope in place or silently drop its metadata. There is no generic purge
command: use a reviewed, audited procedure that also considers replicas and
backups, and validate consistency and explicit unavailable-history behavior
in isolation before resuming service. In particular, removing an unacknowledged
local original can make startup fail closed. No destructive automation.

## Verification boundaries

The [offline workflow](../.github/workflows/offline.yml) runs on the locked
environment with no provider secrets. From the checkout root:

```sh
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyright
env -u EMERGENCE_LIVE uv run --locked pytest -m 'not live'
```

### Explicit opt-in live smoke contract

The root integration owns `tests/live`, the final verification report and
the live-provider matrix. This command is the intended contract, **not**
part of installation or offline CI. Only a separately authorized operator
should run it with real scoped credentials:

```sh
EMERGENCE_LIVE=1 \
EMERGENCE_CONFIG=/absolute/operator-selected.yaml \
EMERGENCE_LIVE_PROVIDERS=comma-configured-names \
EMERGENCE_LIVE_MAX_OUTPUT_TOKENS=8 \
EMERGENCE_LIVE_MAX_REQUEST_USD=0.01 \
EMERGENCE_LIVE_TOTAL_CAP_USD=0.04 \
  uv run pytest tests/live -m live -s
```

Replace `comma-configured-names` with comma-separated names from the selected
settings. If a selected provider has multiple aliases, set
`EMERGENCE_LIVE_MODELS` to a JSON object mapping its configured name to the
chosen alias (for example `{"openai":"chat"}` if those are your configured
names). A selected provider with a single alias needs no override.

The harness uses a fresh temporary migrated DB and local spool, configured
actual prices, verified applicable bounds and scopes, and at most one
inference call per selected provider, with no retry and sanitized output.
Before any call it checks one per-request ceiling for each selected cloud
provider against the total cap and fails if the combined ceilings exceed
that cap. The actual reservation is checked against the per-call ceiling inside
the admission transaction before dispatch; this ceiling remains effective
across calendar renewal and does not alter the USD 100 opening.
Missing or invalid configuration or caps fail an opted-in run;
they are not a pass. Do not run against production DB, spool or NAS state.

Reporting has a separate smoke test and requires explicit
`EMERGENCE_LIVE_REPORTING=1` in addition to live opt-in. Leave that flag
unset for inference only; reporting is then explicitly skipped and
remains unverified. A reporting smoke test is not an account UI check.

Provider model/entitlement, real quota and OpenAI scoped reporting/account
UI, bounded llama.cpp from the target LXC, target systemd/network isolation,
and real NAS interruption/recovery/restore are **not run** by packaging.
They require operator credentials, access and permission; record each
separately in the final matrix. These docs do not establish v0.1 completion.
