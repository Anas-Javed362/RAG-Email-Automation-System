"""
Request ID injection middleware.

Attaches a unique `X-Request-ID` header to every request so that
all log lines for a single request can be correlated together.
Uses UUID4 if the client doesn't supply its own request ID.
"""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logger import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that ensures every request has a unique trace ID.

    - Reads `X-Request-ID` from the incoming request headers if present.
    - Otherwise generates a new UUID4.
    - Attaches it to `request.state.request_id` for downstream use.
    - Echoes the ID back in the response headers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Accept a client-supplied ID or mint a fresh one
        request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))
        request.state.request_id = request_id

        logger.debug(f"Incoming {request.method} {request.url.path} | request_id={request_id}")

        response: Response = await call_next(request)

        # Echo the request ID back so clients can correlate responses
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
