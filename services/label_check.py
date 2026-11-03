"""Read the label off a generated product image and check the brand name on it."""
import base64
import logging
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, Field

from services import creative

logger = logging.getLogger(__name__)

LABEL_MODEL = creative.PLANNING_MODEL  # same gpt-4o-mini pricing as planning_cost
# Spend-cap reserve per check. A 1024px image at detail "high" is about 25k input tokens on
# gpt-4o-mini (~$0.004); the real cost from the usage is what goes on the asset.
RESERVE_USD = Decimal("0.005")
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'"})

# The expected brand name is never sent: told what to look for, a vision model tends to
# "read" the right spelling even where the image is garbled.
PROMPT = (
    "Transcribe all text printed on the product's label in this image, exactly as it appears, "
    "in reading order, character for character, including accents and apostrophes. "
    "Do not correct spelling or guess: write ? for any character you can't read. "
    "Return an empty string if no label text is legible."
)


class LabelReading(BaseModel):
    label_text: str = Field(description="The label text exactly as printed, or empty if none is legible.")


@dataclass
class LabelCheck:
    ok: bool
    text_read: str | None  # None when the check itself failed
    cost_usd: Decimal


def _tokens(text: str) -> list[str]:
    return unicodedata.normalize("NFC", text).translate(APOSTROPHES).casefold().split()


def brand_matches(label_text: str, brand_name: str) -> bool:
    """True if the brand name appears on the label as whole words, ignoring case and apostrophe style."""
    brand, label = _tokens(brand_name), _tokens(label_text)
    if not brand:
        return False
    return any(label[i:i + len(brand)] == brand for i in range(len(label) - len(brand) + 1))


def check_label(image: bytes, content_type: str, brand_name: str) -> LabelCheck:
    """Never raises: a failed check comes back as not ok, so the asset still needs review."""
    data_url = f"data:{content_type};base64,{base64.b64encode(image).decode()}"
    try:
        reading, completion = creative.client.chat.completions.create_with_completion(
            model=LABEL_MODEL,
            response_model=LabelReading,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            ]}],
            max_tokens=300,
            max_retries=1,
        )
    except Exception as exc:  # the image is already paid for: flag it, don't lose it
        logger.warning("Label check failed: %s", type(exc).__name__)
        return LabelCheck(ok=False, text_read=None, cost_usd=Decimal("0"))
    usage = completion.usage
    cost = creative.planning_cost(usage.prompt_tokens, usage.completion_tokens)
    ok = brand_matches(reading.label_text, brand_name)
    if not ok:
        logger.info("Label mismatch: expected %r, read %r", brand_name, reading.label_text)
    return LabelCheck(ok=ok, text_read=reading.label_text, cost_usd=cost)
