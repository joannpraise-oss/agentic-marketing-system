import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from typing import Callable

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.higgsfield.ai"
PROVIDER = "higgsfield"
MODEL = "higgsfield-ai/soul/v2/standard"
IMAGE_PARAMS = {"aspect_ratio": "1:1", "resolution": "1080p", "num_images": 1}
# Physical products with a reference photo. enhance_prompt=False is direct mode: on would force quality=high.
PRODUCT_MODEL = "marketing-studio/image"
PRODUCT_IMAGE_PARAMS = {"aspect_ratio": "1:1", "resolution": "1k", "quality": "low", "enhance_prompt": False}

HTTP_TIMEOUT_SECONDS = 30
GENERATION_TIMEOUT_SECONDS = 120
# Polling schedule from the Higgsfield docs: start at 2s, x1.5 per poll, cap at 10s, plus jitter.
POLL_START_SECONDS = 2.0
POLL_MAX_SECONDS = 10.0
POLL_JITTER_SECONDS = 0.5
TERMINAL_STATUSES = {"completed", "failed", "nsfw", "canceled"}


class HiggsfieldError(Exception):
    """The provider failed or returned something unusable. Messages never contain credentials."""


class HiggsfieldTimeout(HiggsfieldError):
    """The job did not reach a terminal status within the generation timeout."""


class HiggsfieldContentFiltered(HiggsfieldError):
    """Higgsfield's moderation rejected the prompt or the image (status "nsfw"). Not charged."""


class HiggsfieldAuthError(HiggsfieldError):
    """Missing or rejected credentials: our problem, not the caller's."""


@dataclass
class GeneratedImage:
    data: bytes
    content_type: str
    request_id: str


def _auth_headers() -> dict[str, str]:
    # HIGGSFIELD_API_KEY holds "<key_id>:<key_secret>", as the Authorization header expects.
    key = os.getenv("HIGGSFIELD_API_KEY", "")
    if ":" not in key:
        raise HiggsfieldAuthError("Higgsfield is not configured (HIGGSFIELD_API_KEY must be '<key_id>:<key_secret>').")
    return {"Authorization": f"Key {key}"}


def _call(method: str, path: str, **kwargs) -> dict:
    headers = {**_auth_headers(), **kwargs.pop("headers", {})}
    try:
        response = httpx.request(
            method, f"{BASE_URL}{path}", headers=headers, timeout=HTTP_TIMEOUT_SECONDS, **kwargs
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403):
            raise HiggsfieldAuthError(f"Higgsfield rejected our credentials ({status}).") from None
        logger.error("Higgsfield %s %s returned status %s", method, path, status)
        raise HiggsfieldError(f"Higgsfield returned status {status}.") from None
    except httpx.TimeoutException:
        raise HiggsfieldError("Higgsfield did not respond in time.") from None
    except httpx.HTTPError as exc:
        logger.error("Higgsfield %s %s failed: %s", method, path, type(exc).__name__)
        raise HiggsfieldError("Could not reach Higgsfield.") from None
    except ValueError:
        raise HiggsfieldError("Higgsfield returned invalid JSON.") from None


def estimate_cost() -> Decimal:
    """USD cost of one image with IMAGE_PARAMS. Higgsfield only charges completed jobs."""
    # From a live POST /estimate/higgsfield-ai/soul/v2/standard check on 2026-10-21
    # (1:1, 1080p, 1 image): usd "0.006", credits "0.090". Re-check if IMAGE_PARAMS change.
    return Decimal("0.006")


# Marketing Studio's price depends only on prompt length, not format. Live POST
# /estimate/marketing-studio/image checks on 2026-11-03 (PRODUCT_IMAGE_PARAMS, one reference):
# 1000 chars $0.035, 1500 $0.037, 1897 $0.039, 2398 $0.042, 3795 $0.049, about
# $0.0297 + $0.0000051 per character. These round up slightly, so the cost never undercounts.
PRODUCT_BASE_USD = Decimal("0.0300")
PRODUCT_USD_PER_CHAR = Decimal("0.0000052")
# Reserved before planning, when the prompts aren't written yet. Prompts with every appended rule
# run ~1900-2100 characters; a longer one is re-checked against the caps once it's known.
RESERVE_PROMPT_CHARS = 3000


def product_image_cost(prompt: str) -> Decimal:
    """USD cost of one Marketing Studio image for this prompt, rounded up to 4 places."""
    return (PRODUCT_BASE_USD + PRODUCT_USD_PER_CHAR * len(prompt)).quantize(Decimal("0.0001"), ROUND_CEILING)


def estimate_product_image_cost() -> Decimal:
    """What to reserve per Marketing Studio image before the prompt is known."""
    return product_image_cost("x" * RESERVE_PROMPT_CHARS)


def _cancel_quietly(request_id: str) -> None:
    # Only queued jobs can be canceled (and refunded); anything else is best-effort.
    try:
        _call("POST", f"/requests/{request_id}/cancel")
    except HiggsfieldError:
        pass


def _download(url: str) -> tuple[bytes, str]:
    # Output URLs are public CDN links: never send our credentials there.
    try:
        response = httpx.get(url, timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("Higgsfield image download failed: %s", type(exc).__name__)
        raise HiggsfieldError("Could not download the generated image.") from None
    content_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
    return response.content, content_type


def generate_image(
    prompt: str,
    *,
    timeout: float = GENERATION_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> GeneratedImage:
    """Submit a Soul v2 job, poll until it finishes, and return the image bytes."""
    return _generate(MODEL, {"prompt": prompt, **IMAGE_PARAMS}, timeout, sleep, clock)


def generate_product_image(
    prompt: str,
    reference_url: str,
    *,
    timeout: float = GENERATION_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> GeneratedImage:
    """Generate from a product photo. reference_url carries a token: never log it."""
    body = {"prompt": prompt, "image_urls": [reference_url], **PRODUCT_IMAGE_PARAMS}
    return _generate(PRODUCT_MODEL, body, timeout, sleep, clock)


def _generate(
    model: str, body: dict, timeout: float, sleep: Callable[[float], None], clock: Callable[[], float]
) -> GeneratedImage:
    job = _call("POST", f"/{model}", json=body, headers={"Idempotency-Key": uuid.uuid4().hex})
    request_id = job.get("request_id")
    if not request_id:
        raise HiggsfieldError("Higgsfield did not return a request id.")

    deadline = clock() + timeout
    delay = POLL_START_SECONDS
    status = job
    while status.get("status") not in TERMINAL_STATUSES:
        remaining = deadline - clock()
        if remaining <= 0:
            _cancel_quietly(request_id)
            logger.warning("Higgsfield job %s timed out after %ss", request_id, timeout)
            raise HiggsfieldTimeout(f"Image generation did not finish within {timeout:.0f}s.")
        sleep(min(delay + random.uniform(0, POLL_JITTER_SECONDS), remaining))
        delay = min(delay * 1.5, POLL_MAX_SECONDS)
        status = _call("GET", f"/requests/{request_id}/status")

    if status["status"] == "nsfw":
        # Higgsfield's billing docs: a request that ends as nsfw is not charged; reserved credits are refunded.
        logger.warning("Higgsfield job %s rejected by the content filter (nsfw); charged: no (refunded)", request_id)
        raise HiggsfieldContentFiltered("Image generation ended as 'nsfw'.")
    if status["status"] != "completed":
        logger.warning("Higgsfield job %s ended as %s: %s", request_id, status["status"], status.get("error"))
        raise HiggsfieldError(f"Image generation ended as '{status['status']}'.")

    images = status.get("images") or []
    if not images or not images[0].get("url"):
        raise HiggsfieldError("Higgsfield completed without an image URL.")

    data, content_type = _download(images[0]["url"])
    return GeneratedImage(data=data, content_type=content_type, request_id=request_id)
