"""Read product labels: once off the uploaded photo (the reference), then off each generated image."""
import base64
import io
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from difflib import SequenceMatcher

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from services import creative

logger = logging.getLogger(__name__)

LABEL_MODEL = creative.PLANNING_MODEL  # same gpt-4o-mini pricing as planning_cost
# Spend-cap reserve per call. A 1024px image at detail "high" is about 25k input tokens on
# gpt-4o-mini (~$0.004); the real cost from the usage is what goes on the asset.
RESERVE_USD = Decimal("0.005")
# One check: the product locate plus up to two reads.
CHECK_RESERVE_USD = 3 * RESERVE_USD
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'"})
WORD_PUNCTUATION = ".,;:!()[]{}\"“”«»"
# The reader's marks for text it can see but not read: a whole run, or single characters.
UNREADABLE_MARK = re.compile(r"\[\s*unreadable[^\]]*\]", re.IGNORECASE)
UNREADABLE_CHAR = "?"
UNCERTAIN_CHARS = "[]"  # the reader also brackets letters it isn't sure of: "[b]EAUTY"
# How close a word read off the reference photo must be to brand_name to count as a misread of it.
BRAND_SIMILARITY = 0.6

# The expected text is never sent: told what to look for, a vision model tends to "read"
# the right spelling even where the image is garbled.
PROMPT = (
    "Transcribe all text printed on the product's label in this image, exactly as it appears, "
    "in reading order, character for character, including accents and apostrophes. "
    "Do not correct spelling or guess: write ? for any character you can't make out, and write "
    "[unreadable] in place of any text that is there but too small or blurry to read at all. "
    "Leave out text that is covered (by a hand, for example) or turned out of view. "
    "Return an empty string if no label text is visible."
)


# Small label text is only a few pixels tall at true size, and the reader sees the whole image
# downscaled, so it missed the brand's apostrophe. Both reads get the product cropped and
# upscaled so its long side is CROP_LONG_SIDE_PX: at detail "high" that is 2-4 tiles, no more
# than the full image cost.
CROP_LONG_SIDE_PX = 1024
MAX_UPSCALE = 4
MIN_BOX_AREA = 0.005  # a smaller box is a bad guess: read the whole image instead
# Asked for coordinates, gpt-4o-mini put every box near the centre wherever the bottle was. A
# labelled grid drawn on the image works (at detail "high"; "low" and a 512px copy were unreliable),
# though it can miss by a column, so the crop adds half a cell on every side.
GRID = 6
GRID_ROWS = "ABCDEF"
GRID_COLOR = (255, 0, 255)
CELL_PATTERN = re.compile(rf"^([{GRID_ROWS}])([1-{GRID}])$")
LOCATE_PROMPT = (
    f"The image has a {GRID}x{GRID} grid drawn on it; each cell is labeled with its row letter (A at the top) "
    "and column number (1 at the left). List every cell that contains any part of the product's label "
    "(the printed text and logo on the bottle, jar or package). Return an empty list if there is no product."
)


class LabelCells(BaseModel):
    cells: list[str] = Field(description="Grid cells such as \"D3\" that contain part of the product's label.")


@dataclass
class ProductBox:
    """A region as fractions of the image width and height."""
    x0: float
    y0: float
    x1: float
    y1: float


def cells_to_box(cells: list[str]) -> ProductBox | None:
    """The cells' bounding box plus half a cell on every side; None if no cell is valid."""
    found = [m.groups() for m in (CELL_PATTERN.match(c.strip().upper()) for c in cells) if m]
    if not found:
        return None
    rows = [GRID_ROWS.index(r) for r, _ in found]
    cols = [int(c) - 1 for _, c in found]
    half = 0.5 / GRID
    return ProductBox(x0=min(cols) / GRID - half, y0=min(rows) / GRID - half,
                      x1=(max(cols) + 1) / GRID + half, y1=(max(rows) + 1) / GRID + half)


def _with_grid(image: bytes) -> bytes | None:
    """The image as PNG with the labelled locate grid drawn on it; None if it isn't an image."""
    try:
        picture = Image.open(io.BytesIO(image)).convert("RGB")
    except Exception:
        return None
    draw = ImageDraw.Draw(picture)
    width, height = picture.size
    font = ImageFont.load_default(size=max(14, width // 40))
    for i in range(1, GRID):
        draw.line([(width * i // GRID, 0), (width * i // GRID, height)], fill=GRID_COLOR, width=3)
        draw.line([(0, height * i // GRID), (width, height * i // GRID)], fill=GRID_COLOR, width=3)
    for row in range(GRID):
        for col in range(GRID):
            draw.text((width * col // GRID + 6, height * row // GRID + 4), f"{GRID_ROWS[row]}{col + 1}",
                      fill=GRID_COLOR, font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    out = io.BytesIO()
    picture.save(out, "PNG")
    return out.getvalue()


class LabelReading(BaseModel):
    label_text: str = Field(description="The label text exactly as printed, or empty if none is legible.")


MATCH, MISMATCH, UNREADABLE, FAILED = "match", "mismatch", "unreadable", "failed"
# asset.label_outcome's CHECK allows only these; a failed check is stored as NULL (with label_ok False).
STORED_OUTCOMES = (MATCH, UNREADABLE, MISMATCH)


@dataclass
class LabelCheck:
    ok: bool
    text_read: str | None  # None when the read itself failed
    cost_usd: Decimal
    mismatched: list[str] = field(default_factory=list)  # e.g. "brand: expected 'J’aduné'", "extra: puré"
    read_twice: bool = False  # the first read didn't pass, so the image was read again
    second_read: str | None = None  # None when not taken, or when it failed
    # match: every word read exactly. unreadable: nothing visible is wrong, but some text is covered,
    # out of view or too small to read (passes). mismatch: visible text is wrong.
    # failed: no read worked.
    outcome: str | None = None
    cropped: bool = False  # the reads saw the product cropped and upscaled, not the whole image

    def __post_init__(self):
        if self.outcome is None:
            self.outcome = MATCH if self.ok else MISMATCH

    @property
    def reads_logged(self) -> str | None:
        """What goes in asset.label_read: the first read, then the second if one was taken."""
        reads = [self.text_read, self.second_read] if self.read_twice else [self.text_read]
        if all(r is None for r in reads):
            return None
        return SECOND_READ_SEPARATOR.join(FAILED_READ if r is None else r for r in reads)

    @property
    def stored_outcome(self) -> str | None:
        """What goes in asset.label_outcome: match, unreadable or mismatch; None when no read worked."""
        return self.outcome if self.outcome in STORED_OUTCOMES else None


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


def readable_tokens(text: str) -> tuple[list[str], bool]:
    """The words read in full, with their punctuation, and whether any text was marked unreadable."""
    tokens = _tokens(UNREADABLE_MARK.sub(" ", text))
    readable = [t for t in tokens if not any(c in t for c in UNREADABLE_CHAR + UNCERTAIN_CHARS)]
    return readable, bool(UNREADABLE_MARK.search(text)) or len(readable) < len(tokens)


def label_differences(reference: str, label_text: str) -> list[str]:
    """Readable words that aren't on the reference, punctuation included; order and case are ignored.

    A reference word that wasn't read is never a difference: it may be covered, turned out of
    view or too small to read. Only visible text that is wrong counts.
    """
    want, got = Counter(readable_tokens(reference)[0]), Counter(readable_tokens(label_text)[0])
    return [f"extra: {w}" for w in sorted((got - want).elements())]


def missing_words(reference: str, label_text: str) -> bool:
    """True if a reference word wasn't read in full: the label isn't wholly visible and readable."""
    want, got = Counter(readable_tokens(reference)[0]), Counter(readable_tokens(label_text)[0])
    return bool(want - got)


def garbled_brand(label_text: str, brand_name: str) -> bool:
    """True if the brand shows but reads wrong: a run of words close to brand_name that isn't it.

    No such run means the brand is covered or out of view, which passes. Uncertain characters
    count against it ("J'ADUN?"): key text may be covered, never garbled.
    """
    if brand_matches(label_text, brand_name):
        return False
    tokens, size = label_text.split(), len(brand_name.split())
    target = " ".join(_words(brand_name))
    return any(SequenceMatcher(None, " ".join(_words(" ".join(tokens[i:i + size]))), target).ratio()
               >= BRAND_SIMILARITY for i in range(len(tokens) - size + 1))


def _data_url(image: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(image).decode()}"


def locate_product(image: bytes) -> tuple[ProductBox | None, Decimal]:
    """Where the product's label is, roughly, and the call's cost. Never raises: (None, cost) if not found."""
    gridded = _with_grid(image)
    if gridded is None:
        return None, Decimal("0")
    try:
        located, completion = creative.client.chat.completions.create_with_completion(
            model=LABEL_MODEL,
            response_model=LabelCells,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": LOCATE_PROMPT},
                {"type": "image_url", "image_url": {"url": _data_url(gridded, "image/png"), "detail": "high"}},
            ]}],
            max_tokens=100,
            max_retries=1,
        )
    except Exception as exc:
        logger.warning("Product locate failed: %s", type(exc).__name__)
        return None, Decimal("0")
    usage = completion.usage
    return cells_to_box(located.cells), creative.planning_cost(usage.prompt_tokens, usage.completion_tokens)


def crop_product(image: bytes, box: ProductBox) -> bytes | None:
    """The box, upscaled for reading, as PNG; None if the box or the image is unusable."""
    x0, x1 = sorted((min(max(box.x0, 0.0), 1.0), min(max(box.x1, 0.0), 1.0)))
    y0, y1 = sorted((min(max(box.y0, 0.0), 1.0), min(max(box.y1, 0.0), 1.0)))
    if (x1 - x0) * (y1 - y0) < MIN_BOX_AREA:
        return None
    try:
        picture = Image.open(io.BytesIO(image))
        picture.load()
    except Exception:
        return None
    width, height = picture.size
    crop = picture.convert("RGB").crop((int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)))
    scale = min(CROP_LONG_SIDE_PX / max(crop.size), MAX_UPSCALE)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    out = io.BytesIO()
    crop.save(out, "PNG")
    return out.getvalue()


def read_label(image: bytes, content_type: str) -> tuple[str | None, Decimal]:
    """The label text and the read's cost. Never raises: a failed read returns (None, 0)."""
    data_url = _data_url(image, content_type)
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


def _brand_problem(text: str, brand_name: str, brand_required: bool) -> list[str]:
    """A garbled brand, or, where the format must show it, a brand that isn't read exactly."""
    if garbled_brand(text, brand_name) or (brand_required and not brand_matches(text, brand_name)):
        return [f"brand: expected {brand_name!r}"]
    return []


def _wrong(text: str, brand_name: str, reference: str | None, brand_required: bool = False) -> list[str]:
    """What visibly differs: a garbled (or required but missing) brand, and words not on the reference."""
    wrong = _brand_problem(text, brand_name, brand_required)
    return wrong + (label_differences(reference, text) if reference else [])


def assess(
    text: str | None, brand_name: str, reference: str | None, brand_required: bool = False
) -> tuple[str, list[str]]:
    """One read's outcome and what differs.

    Only visible text that is wrong is a mismatch. Text that is covered, out of view or too small
    to read passes as unreadable; match means every word, brand included, was read exactly.
    brand_required: the format must show the brand (studio_hero, model_with_product), so a brand
    that is covered or out of view is a mismatch too.
    """
    if text is None:
        return FAILED, []
    mismatched = _wrong(text, brand_name, reference, brand_required)
    if mismatched:
        return MISMATCH, mismatched
    complete = (brand_matches(text, brand_name) and not readable_tokens(text)[1]
                and not (reference and missing_words(reference, text)))
    return (MATCH if complete else UNREADABLE), []


def combine_reads(
    first: str, second: str, brand_name: str, reference: str | None, brand_required: bool = False
) -> tuple[str, list[str]]:
    """The outcome of two reads that both found a problem.

    A wrong word only counts if both reads saw the same one: on garbled small print the reads
    guess differently ("POTTO INTEIRO", then "POTION INEDITE"), which means it can't be read.
    A word left out never counts (see assess). The brand only counts as wrong (garbled, or
    missing where it is required) if both reads found it wrong.
    """
    wrong_in_both = (_brand_problem(first, brand_name, brand_required)
                     and _brand_problem(second, brand_name, brand_required))
    mismatched = [f"brand: expected {brand_name!r}"] if wrong_in_both else []
    if reference:
        want = Counter(readable_tokens(reference)[0])
        extra_first = Counter(readable_tokens(first)[0]) - want
        extra_second = Counter(readable_tokens(second)[0]) - want
        mismatched += [f"extra: {w}" for w in sorted((extra_first & extra_second).elements())]
    return (MISMATCH, mismatched) if mismatched else (UNREADABLE, [])


def check_label(
    image: bytes, content_type: str, brand_name: str, reference: str | None = None, brand_required: bool = False
) -> LabelCheck:
    """Check a generated image's label: match, unreadable (passes) or mismatch (flagged).

    Covered or out-of-view text passes as unreadable; only visible text that is wrong is flagged,
    and, with brand_required, a brand that doesn't show.

    The reader misreads small text (it dropped a clearly printed apostrophe), so a read that
    disagrees, or fails, is retried once, and the image is flagged only if both reads find the
    same problem (see combine_reads). Both reads see the product cropped and upscaled; the whole
    image if it can't be found.
    Never raises: failed reads come back as not ok, so the asset still needs review.
    """
    box, cost = locate_product(image)
    crop = crop_product(image, box) if box else None
    if crop:
        image, content_type = crop, "image/png"
    text, read_cost = read_label(image, content_type)  # the image is already paid for: flag it, don't lose it
    cost += read_cost
    outcome, mismatched = assess(text, brand_name, reference, brand_required)
    if outcome in (MATCH, UNREADABLE):
        return LabelCheck(ok=True, text_read=text, cost_usd=cost, outcome=outcome, cropped=bool(crop))
    second, second_cost = read_label(image, content_type)
    second_outcome, second_mismatched = assess(second, brand_name, reference, brand_required)
    if second_outcome in (MATCH, UNREADABLE):
        final, mismatched = second_outcome, []
    elif text is not None and second is not None:
        final, mismatched = combine_reads(text, second, brand_name, reference, brand_required)
    else:
        final = MISMATCH if MISMATCH in (outcome, second_outcome) else FAILED
        mismatched = second_mismatched or mismatched
    ok = final in (MATCH, UNREADABLE)
    if not ok:
        logger.info("Label %s: read %r, then %r; %s", final, text, second, mismatched)
    return LabelCheck(ok=ok, text_read=text, cost_usd=cost + second_cost, mismatched=mismatched,
                      read_twice=True, second_read=second, outcome=final, cropped=bool(crop))
