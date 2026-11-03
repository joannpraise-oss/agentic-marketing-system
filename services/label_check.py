"""Read product labels: once off the uploaded photo (the reference), then off each generated image."""
import base64
import logging
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from services import creative

logger = logging.getLogger(__name__)

LABEL_MODEL = creative.PLANNING_MODEL  # same gpt-4o-mini pricing as planning_cost
# Spend-cap reserve per read. A 1024px image at detail "high" is about 25k input tokens on
# gpt-4o-mini (~$0.004); the real cost from the usage is what goes on the asset.
RESERVE_USD = Decimal("0.005")
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'"})
WORD_PUNCTUATION = ".,;:!()[]{}\"“”«»"
# How close a word read off the reference photo must be to brand_name to count as a misread of it.
BRAND_SIMILARITY = 0.6

# The expected text is never sent: told what to look for, a vision model tends to "read"
# the right spelling even where the image is garbled.
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
    text_read: str | None  # None when the read itself failed
    cost_usd: Decimal
    mismatched: list[str] = field(default_factory=list)  # e.g. "missing: peptide", "extra: puré"
    read_twice: bool = False  # the first read didn't pass, so the image was read again
    second_read: str | None = None  # None when not taken, or when it failed

    @property
    def reads_logged(self) -> str | None:
        """What goes in asset.label_read: the first read, then the second if one was taken."""
        reads = [self.text_read, self.second_read] if self.read_twice else [self.text_read]
        if all(r is None for r in reads):
            return None
        return SECOND_READ_SEPARATOR.join(FAILED_READ if r is None else r for r in reads)


SECOND_READ_SEPARATOR = "\n--- second read ---\n"
FAILED_READ = "(read failed)"


def _tokens(text: str) -> list[str]:
    return unicodedata.normalize("NFC", text).translate(APOSTROPHES).casefold().split()


def _words(text: str) -> list[str]:
    return [w for w in (t.strip(WORD_PUNCTUATION) for t in _tokens(text)) if w]


def brand_matches(label_text: str, brand_name: str) -> bool:
    """True if the brand name appears on the label as whole words, ignoring case and apostrophe style."""
    brand, label = _tokens(brand_name), _tokens(label_text)
    if not brand:
        return False
    return any(label[i:i + len(brand)] == brand for i in range(len(label) - len(brand) + 1))


def reference_label(label_text: str, brand_name: str) -> str:
    """The reference label with its brand spelled as brand_name.

    The uploaded photo can be soft enough that its brand reads wrong (J’aduné read as TADUNÉ),
    and brand_name is the authority. The closest run of words is replaced; the rest is kept.
    """
    tokens, brand = label_text.split(), brand_name.split()
    if not brand or brand_matches(label_text, brand_name):
        return label_text
    target = " ".join(_words(brand_name))
    windows = range(len(tokens) - len(brand) + 1)
    scored = [(SequenceMatcher(None, " ".join(_words(" ".join(tokens[i:i + len(brand)]))), target).ratio(), i)
              for i in windows]
    if not scored or max(scored)[0] < BRAND_SIMILARITY:
        return label_text
    _, start = max(scored)
    return " ".join(tokens[:start] + brand + tokens[start + len(brand):])


def label_differences(reference: str, label_text: str) -> list[str]:
    """Words missing from, or added to, the label compared with the reference; order is ignored."""
    want, got = Counter(_words(reference)), Counter(_words(label_text))
    return ([f"missing: {w}" for w in sorted((want - got).elements())]
            + [f"extra: {w}" for w in sorted((got - want).elements())])


def read_label(image: bytes, content_type: str) -> tuple[str | None, Decimal]:
    """The label text and the read's cost. Never raises: a failed read returns (None, 0)."""
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
    except Exception as exc:
        logger.warning("Label read failed: %s", type(exc).__name__)
        return None, Decimal("0")
    usage = completion.usage
    return reading.label_text, creative.planning_cost(usage.prompt_tokens, usage.completion_tokens)


def _mismatches(text: str | None, brand_name: str, reference: str | None) -> list[str] | None:
    """What's wrong with one read, [] if it passes; None if the read failed."""
    if text is None:
        return None
    mismatched = [] if brand_matches(text, brand_name) else [f"brand: expected {brand_name!r}"]
    if reference:
        mismatched += label_differences(reference, text)
    return mismatched


def check_label(image: bytes, content_type: str, brand_name: str, reference: str | None = None) -> LabelCheck:
    """Check a generated image's label: brand_name must be on it, and with a reference, every word must match.

    The reader misreads small text (it dropped a clearly printed apostrophe), so a read that
    disagrees, or fails, is retried once, and the image is flagged only if both reads disagree.
    Never raises: failed reads come back as not ok, so the asset still needs review.
    """
    text, cost = read_label(image, content_type)  # the image is already paid for: flag it, don't lose it
    if _mismatches(text, brand_name, reference) == []:
        return LabelCheck(ok=True, text_read=text, cost_usd=cost)
    second, second_cost = read_label(image, content_type)
    first_wrong, second_wrong = _mismatches(text, brand_name, reference), _mismatches(second, brand_name, reference)
    ok = second_wrong == []
    mismatched = [] if ok else (second_wrong or first_wrong or [])
    if not ok:
        logger.info("Label mismatch: read %r, then %r; %s", text, second, mismatched)
    return LabelCheck(ok=ok, text_read=text, cost_usd=cost + second_cost, mismatched=mismatched,
                      read_twice=True, second_read=second)
