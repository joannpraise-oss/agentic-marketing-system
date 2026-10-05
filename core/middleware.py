import logging
import time
import uuid

from fastapi import Request

from core.errors import internal_error_response
from core.log_config import request_id_var

logger = logging.getLogger(__name__)


async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            # Handled here, inside CORSMiddleware, so the 500 still carries CORS headers
            # and the browser shows the real error instead of a CORS failure.
            logger.exception("Unhandled error: %s %s", request.method, request.url.path)
            response = internal_error_response(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s in %.1fms", request.method, request.url.path, response.status_code, duration_ms
        )
        return response
    finally:
        request_id_var.reset(token)
