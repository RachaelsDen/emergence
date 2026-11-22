from collections.abc import AsyncIterator

import httpx
import pytest

from emergence.resources.reporting_http import ReportingHTTP
from emergence.resources.schemas import ReportingError, ReportingIssue

URL = "https://reporting.invalid/quota"


@pytest.mark.anyio
@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
async def test_reporting_rejects_encoding_before_reading_body(encoding: str) -> None:
    class UnreadableBody(httpx.AsyncByteStream):
        def __aiter__(self) -> AsyncIterator[bytes]:
            raise AssertionError("an encoded body must not be read")

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers.get_list("accept-encoding") == ["identity"]
        return httpx.Response(200, headers={"Content-Encoding": encoding}, stream=UnreadableBody())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        reporting = ReportingHTTP(client, timeout=1, limit=1024)
        with pytest.raises(ReportingError) as caught:
            await reporting.get(URL, headers={"accept-encoding": "gzip"})
    assert caught.value.issue is ReportingIssue.TRANSPORT


@pytest.mark.anyio
async def test_reporting_stops_reading_when_raw_chunks_exceed_remaining_budget() -> None:
    class ChunkedBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'{"quota":'
            yield b"10}"
            raise AssertionError("must stop at the first chunk exceeding the remaining budget")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=ChunkedBody()))
    ) as client:
        reporting = ReportingHTTP(client, timeout=1, limit=1024)
        with pytest.raises(ReportingError) as caught:
            await reporting.get(URL, headers={}, remaining=10)
    assert caught.value.issue is ReportingIssue.TOO_LARGE
