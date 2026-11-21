"""Stable gateway failures shared by admission and execution."""


class GatewayError(Exception):
    def __init__(self, code: str, status: int = 503, request_id: str | None = None) -> None:
        self.code = code
        self.status = status
        self.request_id = request_id
        super().__init__(code)
