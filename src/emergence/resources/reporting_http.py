"""Bounded GET transport for reporting only; never forwards redirects or client auth."""

from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from anyio import fail_after

from .reporting_parsers import WireObject, decode_body
from .schemas import ReportingError, ReportingIssue


@dataclass(frozen=True, slots=True)
class ReportingHTTP:
    client: httpx.AsyncClient
    timeout: float
    limit: int

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: Sequence[tuple[str, str]] = (),
        remaining: int | None = None,
    ) -> tuple[WireObject, int]:
        body = bytearray()
        limit = self.limit if remaining is None else remaining
        try:
            with fail_after(self.timeout):
                async with self.client.stream(
                    "GET",
                    url,
                    headers=headers,
                    params=tuple(params),
                    auth=None,
                    follow_redirects=False,
                    timeout=self.timeout,
                ) as response:
                    match response.status_code:
                        case 401 | 403:
                            raise ReportingError(ReportingIssue.UNAUTHORIZED)
                        case 429:
                            raise ReportingError(ReportingIssue.RATE_LIMITED)
                        case code if not 200 <= code < 300:
                            raise ReportingError(ReportingIssue.HTTP_ERROR)
                        case _:
                            pass
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        if len(body) + len(chunk) > limit:
                            raise ReportingError(ReportingIssue.TOO_LARGE)
                        body.extend(chunk)
        except (TimeoutError, httpx.TimeoutException):
            raise ReportingError(ReportingIssue.TIMEOUT) from None
        except (httpx.RequestError, httpx.InvalidURL, UnicodeError, ValueError):
            raise ReportingError(ReportingIssue.TRANSPORT) from None
        return decode_body(bytes(body)), len(body)
