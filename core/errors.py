import logging

import instructor.core.exceptions as instructor_exc
import openai
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

DEFAULT_RETRY_AFTER = 30


def error_response(
    request: Request,
    status_code: int,
    code: str,
    detail: str,
    headers: dict[str, str] | None = None,
    **extra,
) -> JSONResponse:
    """Every error leaves the API in this one shape: detail, code, request_id."""
    request_id = getattr(request.state, "request_id", None)
    body = {"detail": detail, "code": code, "request_id": request_id, **extra}
    campaign_id = getattr(request.state, "campaign_id", None)
    if campaign_id is not None:
        body["campaign_id"] = campaign_id
    headers = dict(headers or {})
    if request_id:
        headers["X-Request-ID"] = request_id
    return JSONResponse(body, status_code=status_code, headers=headers)


def internal_error_response(request: Request) -> JSONResponse:
    return error_response(
        request, 500, "internal_error", "Something went wrong on our side. Please try again later."
    )


def _retry_after(exc: openai.APIStatusError) -> str:
    value = exc.response.headers.get("retry-after", "")
    return value if value.isdigit() else str(DEFAULT_RETRY_AFTER)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errors = [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
            for e in exc.errors()
        ]
        return error_response(request, 400, "invalid_input", "Invalid request data.", errors=errors)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return error_response(
            request, exc.status_code, f"http_{exc.status_code}", str(exc.detail), headers=exc.headers
        )

    @app.exception_handler(openai.RateLimitError)
    @app.exception_handler(openai.InternalServerError)
    async def ai_busy(request: Request, exc: openai.APIStatusError):
        logger.warning("OpenAI busy (%s): %s", type(exc).__name__, exc)
        return error_response(
            request, 503, "ai_busy", "The AI service is busy. Please retry shortly.",
            headers={"Retry-After": _retry_after(exc)},
        )

    @app.exception_handler(openai.APITimeoutError)
    async def ai_timeout(request: Request, exc: openai.APITimeoutError):
        logger.warning("OpenAI request timed out: %s", exc)
        return error_response(request, 504, "ai_timeout", "The AI service timed out. Please try again.")

    @app.exception_handler(openai.APIConnectionError)
    async def ai_connection(request: Request, exc: openai.APIConnectionError):
        logger.warning("OpenAI connection failed: %s", exc)
        return error_response(request, 502, "ai_unavailable", "Could not reach the AI service.")

    @app.exception_handler(openai.AuthenticationError)
    @app.exception_handler(openai.PermissionDeniedError)
    async def ai_auth(request: Request, exc: openai.APIStatusError):
        # Our credentials problem, not the caller's: log it loudly, say nothing specific.
        logger.critical("OpenAI credentials rejected (%s) - check OPENAI_API_KEY", type(exc).__name__)
        return internal_error_response(request)

    @app.exception_handler(openai.APIError)
    @app.exception_handler(instructor_exc.InstructorRetryException)
    async def ai_bad_response(request: Request, exc: Exception):
        logger.error("AI service returned an unusable response: %s", exc, exc_info=exc)
        return error_response(request, 502, "ai_bad_response", "The AI service returned an invalid response.")

    @app.exception_handler(SQLAlchemyError)
    async def db_error(request: Request, exc: SQLAlchemyError):
        # The request's Session is closed (and rolled back) by get_session on the way out.
        logger.error("Database error: %s", exc, exc_info=exc)
        return error_response(request, 500, "db_error", "A database error occurred. Please try again.")
