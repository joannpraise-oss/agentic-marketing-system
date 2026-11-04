import logging
import os
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

import instructor
from dotenv import load_dotenv
from langsmith import wrappers
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationInfo, model_validator

from core.placeholders import asset_texts, find_placeholders
from schemas.creative import (
    DEFAULT_CREATIVE_COUNT,
    FORMATS_BY_OFFER_TYPE,
    MAX_ON_IMAGE_WORDS,
    MAX_RENDERED_WORDS,
    CreativeFormat,
    CreativeRequest,
    on_image_text_problem,
)

load_dotenv()

logger = logging.getLogger(__name__)

client = instructor.from_openai(wrappers.wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY"))))

PLANNING_MODEL = "gpt-4o-mini"
# gpt-4o-mini list price, USD per 1M tokens. Update if the model or price changes.
INPUT_USD_PER_MTOK = Decimal("0.15")
OUTPUT_USD_PER_MTOK = Decimal("0.60")
# Planning cost is only known afterwards; this reserve goes into the pre-run spend check.
PLANNING_RESERVE_USD = Decimal("0.01")
NUM_CREATIVES = DEFAULT_CREATIVE_COUNT
MAX_CORE_MESSAGE_WORDS = 20
TREND_NOTES_PATH = Path(__file__).resolve().parent.parent / "config" / "trend_notes.md"

# Every image prompt must open with its format's framing, so the label and the picture agree.
FORMAT_OPENERS: dict[str, str] = {
    "ugc": "Candid smartphone photo",
    "model_with_product": "Photo of a fictional model",
    "lifestyle": "Eye-level lifestyle photo",
    "studio_hero": "Studio hero shot",
    "flat_lay": "Top-down flat lay",
    "problem_solution": "Problem-solution photo",
    "seasonal": "Seasonal photo",
    "before_after": "Before-and-after photo",
    "experience": "Experience photo",
    "mood_scene": "Atmospheric mood scene",
    "narrator": "Photo of a fictional narrator",
}
# What each format shows; build_prompt lists these for the formats the offer type allows.
FORMAT_DESCRIPTIONS: dict[str, str] = {
    "ugc": "phone-camera feel, a fictional person using the offer at home",
    "model_with_product": "a fictional model wearing, holding or applying it",
    "lifestyle": "the offer in a styled real-world setting",
    "studio_hero": "clean background, the product as the star",
    "flat_lay": "product and props arranged top-down",
    "problem_solution": "shows the pain point and the offer solving it",
    "seasonal": "tied to a holiday, gifting or a timely moment",
    "before_after": "the same fictional subject before and after, side by side",
    "experience": "the moment of receiving the service (e.g. in the chair) and how it feels",
    "mood_scene": "an atmospheric scene that carries the content's mood",
    "narrator": "a fictional narrator or host figure presenting the content",
}
# Overhead framing is what makes a flat lay; it isn't allowed under any other label.
# ("overhead" alone is left out: "overhead lighting" is common in normal prompts.)
FLAT_LAY_CUES = (
    "flat lay", "flat-lay", "flatlay", "top-down", "top down", "overhead shot", "overhead view",
    "from directly above", "bird's-eye", "bird's eye", "birds-eye",
)


class FAQ(BaseModel):
    question: str
    answer: str


class CreativeItem(BaseModel):
    format: CreativeFormat = Field(description="The creative format this concept uses.")
    rationale: str = Field(description="One sentence on why this format suits this brand.")
    prompt: str = Field(description="Image-generation prompt for one ad creative.")
    caption: str = Field(description="Ad caption in the brand voice.")
    alt_text: str = Field(description="Literal description of the image for screen readers.")
    faq: list[FAQ] = Field(min_length=2, max_length=2)
    angle: str | None = Field(default=None, description="The requested angle this creative is for, copied exactly.")
    on_image_text: str | None = Field(default=None, description="Words shown on the image, if text is used.")
    core_message: str | None = Field(
        default=None, description="Creative tests only: this angle's core message in one line.",
    )
    needs_input: list[str] = Field(
        default_factory=list,
        description="Every [▢ ...] placeholder used in the caption or FAQ, copied exactly.",
    )

    @model_validator(mode="after")
    def placeholders_are_tracked(self) -> "CreativeItem":
        # Placeholders would be drawn into the image or read out by screen readers.
        if find_placeholders([self.prompt]):
            raise ValueError("The image prompt must not contain [▢ ...] placeholders.")
        if find_placeholders([self.alt_text]):
            raise ValueError("alt_text must describe the image and not contain [▢ ...] placeholders.")
        if find_placeholders([self.on_image_text]):
            raise ValueError("on_image_text must not contain [▢ ...] placeholders.")
        opener = FORMAT_OPENERS[self.format]
        if not self.prompt.strip().lower().startswith(opener.lower()):
            raise ValueError(f'A {self.format} prompt must start with "{opener}".')
        if self.format != "flat_lay":
            cue = next((c for c in FLAT_LAY_CUES if c in self.prompt.lower()), None)
            if cue:
                raise ValueError(f'"{cue}" describes a flat lay, not {self.format}. Change the framing or the format.')
        # The text is the source of truth: needs_input always lists exactly what's in it.
        self.needs_input = find_placeholders(asset_texts(self.caption, None, self.faq))
        return self


@dataclass(frozen=True)
class PlanSpec:
    """What a valid plan must look like for one request; passed to validation as context."""
    count: int = NUM_CREATIVES
    offer_type: str = "physical_product"
    angles: tuple[str, ...] | None = None
    text_choice: str = "none"
    text_mode: str = "overlay"
    own_texts: tuple[str, ...] | None = None
    creative_test: bool = False
    test_format: str | None = None
    vary_format: bool = False

    @property
    def single_format_test(self) -> bool:
        return self.creative_test and not self.vary_format

    @classmethod
    def from_request(cls, request: CreativeRequest) -> "PlanSpec":
        angles = tuple(request.angles) if request.angles else None
        own = request.own_texts()
        return cls(
            count=request.creative_count(), offer_type=request.offer_type, angles=angles,
            text_choice=request.text_choice, text_mode=request.text_mode,
            own_texts=tuple(own) if own else None,
            creative_test=request.creative_test, test_format=request.test_format,
            vary_format=request.vary_format,
        )


def creative_count(request: CreativeRequest) -> int:
    return request.creative_count()


def _check_on_image_text(item: "CreativeItem", spec: PlanSpec, index: int) -> None:
    """Fill or validate item.on_image_text for the request's text choice and mode."""
    if spec.text_choice == "none":
        item.on_image_text = None
        return
    if spec.text_choice == "own":
        item.on_image_text = spec.own_texts[index]  # the caller's words, never the model's
    elif not item.on_image_text or not item.on_image_text.strip():
        raise ValueError(f"Creative {index + 1} needs on_image_text (text_choice is 'suggest').")
    else:
        item.on_image_text = item.on_image_text.strip()
        problem = on_image_text_problem(item.on_image_text, spec.text_mode)
        if problem:
            raise ValueError(f"Creative {index + 1}: {problem}")
    quoted = f'"{item.on_image_text}"'
    if spec.text_mode == "rendered":
        if quoted not in item.prompt:
            raise ValueError(f"Creative {index + 1}: a rendered-text prompt must contain the exact text {quoted}.")
    else:
        if "negative space" not in item.prompt.lower():
            raise ValueError(
                f"Creative {index + 1}: an overlay prompt must leave clean negative space for the text (say where)."
            )
        if quoted.lower() in item.prompt.lower():
            raise ValueError(
                f"Creative {index + 1}: overlay text is added later; keep {quoted} out of the image prompt."
            )


class CreativePlan(BaseModel):
    items: list[CreativeItem] = Field(min_length=1)
    format_rationale: str | None = Field(
        default=None, description="Creative tests with one chosen format: why that format suits every angle.",
    )
    shared_composition: str | None = Field(
        default=None,
        description="Creative tests with one format: the framing, setting, lighting and style every prompt repeats word for word.",
    )

    @model_validator(mode="after")
    def matches_spec(self, info: ValidationInfo) -> "CreativePlan":
        # A validation error here makes Instructor re-ask the model.
        spec = (info.context or {}).get("spec") or PlanSpec()
        items = self.items
        if len(items) != spec.count:
            if spec.count == 1:
                raise ValueError(f"Return exactly 1 creative in items, not {len(items)}: one concept, not one per format.")
            raise ValueError(f"Return exactly {spec.count} creatives, not {len(items)}.")
        prompts = {item.prompt.strip().lower() for item in items}
        if len(prompts) != len(items):
            raise ValueError("Each creative needs a distinct image prompt.")
        if spec.angles:
            # One creative per angle, in order; formats may repeat because the angles differ.
            # With one angle there's nothing to match: the model often leaves the optional field
            # empty, which failed every single-angle run. With several, a missing angle is an error.
            if len(spec.angles) == 1 and not (items[0].angle or "").strip():
                items[0].angle = spec.angles[0]
            got = [(item.angle or "").strip().casefold() for item in items]
            if got != [a.casefold() for a in spec.angles]:
                raise ValueError(f"Return one creative per angle, in this order: {list(spec.angles)}.")
            for item, angle in zip(items, spec.angles):
                item.angle = angle
        else:
            if len({item.format for item in items}) != len(items):
                raise ValueError("Each creative must use a different format.")
            for item in items:
                item.angle = None
        allowed = FORMATS_BY_OFFER_TYPE[spec.offer_type]
        for item in items:
            if item.format not in allowed:
                raise ValueError(
                    f"{item.format} does not fit a {spec.offer_type} offer. Use one of: {', '.join(allowed)}."
                )
        for index, item in enumerate(items):
            _check_on_image_text(item, spec, index)
        self._check_creative_test(spec)
        return self

    def _check_creative_test(self, spec: PlanSpec) -> None:
        if not spec.creative_test:
            self.format_rationale = self.shared_composition = None
            for item in self.items:
                item.core_message = None
            return
        for index, item in enumerate(self.items, 1):
            message = (item.core_message or "").strip()
            if not message or "\n" in message or len(message.split()) > MAX_CORE_MESSAGE_WORDS:
                raise ValueError(
                    f"Creative {index}: core_message must be one line of at most {MAX_CORE_MESSAGE_WORDS} words."
                )
            if find_placeholders([message]):
                raise ValueError(f"Creative {index}: core_message must not contain [▢ ...] placeholders.")
            item.core_message = message
        if not spec.single_format_test:
            self.format_rationale = self.shared_composition = None
            return
        # Only the angle may vary: one format and one composition across the whole test.
        formats = {item.format for item in self.items}
        if spec.test_format and formats != {spec.test_format}:
            raise ValueError(f"Every creative in this test must use the format {spec.test_format}.")
        if len(formats) != 1:
            raise ValueError("Every creative in this test must use the same format.")
        if spec.test_format:
            self.format_rationale = None
        elif not (self.format_rationale or "").strip():
            raise ValueError("Explain in format_rationale why this one format suits every angle.")
        composition = (self.shared_composition or "").strip()
        if not composition:
            raise ValueError("Write shared_composition: the framing, setting, lighting and style every prompt shares.")
        for index, item in enumerate(self.items, 1):
            if composition.lower() not in item.prompt.lower():
                raise ValueError(f"Creative {index}: the image prompt must include shared_composition word for word.")
        self.shared_composition = composition


def load_trend_notes() -> str:
    """Contents of config/trend_notes.md, or "" if the file is missing."""
    try:
        return TREND_NOTES_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning("Trend notes file not found at %s", TREND_NOTES_PATH)
        return ""


def planning_cost(prompt_tokens: int, completion_tokens: int) -> Decimal:
    return (
        Decimal(prompt_tokens) * INPUT_USD_PER_MTOK + Decimal(completion_tokens) * OUTPUT_USD_PER_MTOK
    ) / Decimal(1_000_000)


# Formats where a person holds or uses the product: in the live run, a small, angled or
# held bottle garbled its label, so these prompts require a front-facing, legible label. An
# earlier "fills at least 25% of the frame" pushed the bottle toward the lens and oversized it,
# so true size wins over label size.
LABEL_FACING_FORMATS = ("ugc", "model_with_product")
LABEL_FACING_RULE = (
    "The product faces the camera; the label is in focus and legible, and the product stays at "
    "true real-world size, even if that makes the label smaller."
)
# Formats that always show people. lifestyle sometimes does: its prompt is checked for people words.
PEOPLE_FORMATS = ("ugc", "model_with_product", "before_after", "experience", "narrator")
PEOPLE_WORDS = re.compile(
    r"\b(woman|women|man|men|person|people|model|girl|boy|child|children|kids?|couple|friends?|"
    r"she|he|her|his|face|hands?)\b",
    re.IGNORECASE,
)
# The planner ignored REALISM_RULE in the live run, so this short form goes on the prompt itself.
# "Editorial portrait" implied retouching (poreless, glossy skin), so it's a candid phone photo.
# The wardrobe and framing below keep it clear of the content filter, which rejected an earlier
# skin-detail wording; a rejection is retried once with SAFER_REALISM_PROMPT.
REALISM_PROMPT = (
    "Candid smartphone photo, unretouched, no beauty filter, no skin smoothing; visible pores and natural "
    "texture on the face, slight unevenness in skin tone, matte skin with natural shine only on the high "
    "points, natural teeth."
)
WARDROBE_FRAMING = (
    "The person is fully clothed in everyday wear suited to the brand (e.g. a linen shirt or knit top), "
    "framed head-and-shoulders or waist-up."
)
# The retry after a content-filter rejection: no skin wording at all, and risky scene words swapped.
SAFER_REALISM_PROMPT = "Candid smartphone photo, unretouched, no beauty filter, natural light."
DROPPED_PEOPLE_WORDS = re.compile(r",?\s*\b(?:(?:faint|visible|subtle|small)\s+)*veins?\b|,?\s*\bglowing skin\b",
                                  re.IGNORECASE)
SAFER_SWAPS = [
    (re.compile(p, re.IGNORECASE), replacement) for p, replacement in [
        (r"\b(?:sensual|sexy|seductive|intimate|sultry|steamy)\s*", ""),
        (r"\b(?:bare|exposed|naked|nude)\s+(?:skin|shoulders?|arms?|legs?|back)\b", "a knit top"),
        (r"\b(?:tank top|camisole|towel|robe|bathrobe|lingerie|bikini|swimsuit|bra)\b", "linen shirt"),
        (r"\b(?:bathroom|shower|bathtub|bath)\b", "dressing room"),
        (r"\bbedroom\b", "living room"),
        (r"\bskin\b", "look"),
    ]
]
# A live run showed the person holding one bottle and spraying from a second one.
ONE_PRODUCT = "Exactly one product in the image."
ONE_PRODUCT_HELD = "Exactly one product in the image; the person holds and uses the same single product."
# Live runs showed the product oversized next to hands and faces, and blending into the hand.
# Product-only shots get "surroundings" instead of hands, so the rule doesn't add a hand.
SCALE_WITH_SIZE = "The product ({size}) is shown at true real-world size relative to {relative_to}."
SCALE_HAND_LIMIT = "no larger than a hand from wrist to fingertip"
SCALE_DEFAULT = "The product is shown at realistic real-world size, not enlarged."
SEPARATION = (
    "The product is in sharp focus and clearly separated from {from_what} by contrast in tone or lighting, "
    "with a soft highlight along its edge."
)
HOLD_LOWER_HALF = "The hand holds the lower half so the label stays fully visible."


def shows_people(format: str, prompt: str) -> bool:
    return format in PEOPLE_FORMATS or (format == "lifestyle" and bool(PEOPLE_WORDS.search(prompt)))


def product_rules(people: bool, product_size: Optional[str]) -> str:
    """One product, its real size, and its separation from the hand and background."""
    if product_size:
        relative_to = f"the hands and face, {SCALE_HAND_LIMIT}" if people else "its surroundings"
        scale = SCALE_WITH_SIZE.format(size=product_size, relative_to=relative_to)
    else:
        scale = SCALE_DEFAULT
    separation = SEPARATION.format(from_what="the hand and background" if people else "the background")
    rules = [ONE_PRODUCT_HELD if people else ONE_PRODUCT, scale, separation]
    return " ".join(rules + [HOLD_LOWER_HALF] if people else rules)


def safer_scene(prompt: str) -> str:
    """The planner's scene with the words most likely to trip a content filter swapped or removed."""
    for pattern, replacement in SAFER_SWAPS:
        prompt = pattern.sub(replacement, prompt)
    return prompt


def finalize_image_prompt(
    prompt: str, format: str, offer_type: str, brand_name: str, has_photo: bool,
    product_size: Optional[str] = None, safer: bool = False,
) -> str:
    """The prompt the image model gets: the planner's prompt plus the rules it must not drop.

    safer: the wording for the one retry after a content-filter rejection.
    """
    people = shows_people(format, prompt)
    if people:
        prompt = DROPPED_PEOPLE_WORDS.sub("", prompt)
        if safer:
            prompt = safer_scene(prompt)
        prompt = f"{prompt} {SAFER_REALISM_PROMPT if safer else REALISM_PROMPT} {WARDROBE_FRAMING}"
    if offer_type == "physical_product":
        prompt = f"{prompt} {product_rules(people, product_size)}"
    return with_product_label(prompt, brand_name, format) if has_photo else prompt


def with_product_label(prompt: str, brand_name: str, format: str) -> str:
    """Image prompt for a run with a product photo: the model must copy the label, not reinvent it."""
    facing = f" {LABEL_FACING_RULE}" if format in LABEL_FACING_FORMATS else ""
    return (
        f"{prompt}{facing} The product label reads exactly «{brand_name}» with the apostrophe. Keep all label text "
        "exactly as in the reference photo; do not change or invent any text. "
        "Do not add objects beyond those described in this prompt."
    )


def _product_photo_section(request: CreativeRequest) -> str:
    if request.product_photo_path is None:
        return ""
    return (
        "\nPRODUCT PHOTO:\n"
        "A reference photo of the real product goes to the image model with every prompt, and the\n"
        "model copies the product from it. Each prompt describes the scene: setting, lighting,\n"
        "composition, people, props and the product's placement. Refer to the product plainly\n"
        "(e.g. \"the bottle\") and do not redesign it: no new shape, colors, packaging or label.\n"
        "The product's own label and logo stay as in the photo; the no-logos/no-words rule below\n"
        "is about adding anything else.\n"
        "In ugc and model_with_product, the product faces the camera; the label is in focus and\n"
        "legible, never angled away or hidden by a hand, and the product stays at true real-world\n"
        "size. True size wins over label size: never enlarge the product to make the label bigger.\n"
    )


def _props_rule(request: CreativeRequest) -> str:
    if request.offer_type != "physical_product":
        return ""
    return (
        "- Props are allowed only when you deliberately choose them to fit the format, brand and\n"
        "  audience, and name them in the image prompt. Never add props that imply ingredients,\n"
        "  results or features not stated in the brief. The product stays the focal point.\n"
        "- Never include unbranded product containers (bottles, jars, tubes, pumps, droppers,\n"
        "  compacts) that could be mistaken for other products from this brand.\n"
        "- Mood props like candles, flowers, mirrors and fabrics are encouraged for lifestyle and\n"
        "  luxury formats.\n"
    )


# Formats that show people (lifestyle only sometimes). AI images default to flawless, plastic
# skin; these prompts must ask for real, photographed people instead.
REALISM_RULE = """- Realism, in every image prompt that shows people (ugc, model_with_product, lifestyle
  with people, before_after, experience, narrator): People must look like real, photographed
  humans, not AI-perfect. Describe natural skin: visible pores and fine texture, subtle
  unevenness, and where it suits the person, freckles, moles, small spots, or piercings.
  Give skin real depth: natural shadows, highlights and undertones, and varied skin color
  within the face. Avoid airbrushed or plastic skin, perfect
  symmetry, and over-smoothed hair. Vary these details across people and ads; don't give
  everyone the same features. Lighting should look like a real camera: natural falloff, soft
  shadows, slight grain where it fits UGC. People are fully clothed in everyday wear suited
  to the brand (e.g. a linen shirt or knit top), framed head-and-shoulders or waist-up.
"""


def offer_type_guidance(request: CreativeRequest) -> str:
    """Return the OFFER TYPE section of the planner prompt for request.offer_type.

    Must contain:
      - Which offer type this is, in plain words, and what the ad is selling.
      - What the visuals must show for that type:
          physical_product - the product itself, recognisable, in use or on display.
          service          - the result and the experience, not a product: the
                             before -> after, the moment in the chair.
          content          - mood scenes, a narrator, or a quote-card style image
                             (text added via on_image_text, not invented in the image).
      - What to avoid for that type (e.g. no fake product shots for a service,
        no invented book covers or quotes for content).
      - The formats allowed for the type are listed separately by build_prompt
        (FORMATS_BY_OFFER_TYPE), so describe intent, not the format list.

    Hint: a dict keyed by offer type with a few lines each, returned under an
    "OFFER TYPE:" heading, keeps this readable. request.brand.offer is the
    thing being promoted, whatever the type.
    """
    guidance = {
        "physical_product": (
            "This ad sells a physical product. Show the product itself, recognizable,\n"
            "in use or on display. Avoid: a generic stand-in that looks like a different product."
        ),
        "service": (
            "This ad sells a service, not a physical product. Show the RESULT and the\n"
            "EXPERIENCE the customer gets: the finished outcome, the before -> after, the\n"
            "moment in the chair, the client's confidence. Use the way this industry\n"
            "typically shows its work, and lean on the TREND NOTES and STRATEGY NOTES\n"
            "(when given) for current scroll-stopping ideas.\n"
            "Avoid: false claims about results; images unrelated to the service; product\n"
            "shots of items the business does not sell; results that look impossible."
        ),
        "content": (
            "This ad promotes content (a book, tips, quotes, ideas), not a product.\n"
            "Show a visual that makes the message land in one glance: a mood scene, a\n"
            "narrator, or a quote-card style background with space for on-image text.\n"
            "Follow what audiences engage with today, from the TREND NOTES and STRATEGY\n"
            "NOTES (when given), and the formats this kind of content commonly uses.\n"
            "Avoid: quotes or lines not in the source text; attributing words to real\n"
            "people; visuals unrelated to the message; implying endorsements."
        ),

    }
    return f"OFFER TYPE: {request.offer_type}\n{guidance[request.offer_type]}\n"


def _format_menu(offer_type: str) -> str:
    return "\n".join(
        f"- {fmt}: {FORMAT_DESCRIPTIONS[fmt]}\n  (prompt starts with \"{FORMAT_OPENERS[fmt]}\")"
        for fmt in FORMATS_BY_OFFER_TYPE[offer_type]
    )


def _task_intro(request: CreativeRequest) -> str:
    if not request.angles:
        return (
            f"Create exactly {NUM_CREATIVES} ad concepts, each in a DIFFERENT format chosen to fit this\n"
            "brand, goal and platform:"
        )
    angles = "\n".join(f"{i}. {angle}" for i, angle in enumerate(request.angles, 1))
    if len(request.angles) == 1:
        # The plural wording ("exactly 1 ad concepts ... formats may repeat") above a list of
        # formats made the model return one concept per format in live single-angle runs.
        return (
            "Create exactly ONE ad concept (items holds a single item) for the angle below.\n"
            "Pick the ONE format from the list that best fits this angle, brand, goal and platform.\n"
            "Do not make a concept for each format.\n"
            f"ANGLE: {request.angles[0]}\n\nFormats:"
        )
    if not request.creative_test:
        return (
            f"Create exactly {len(request.angles)} ad concepts for the same offer, one per angle below, in this\n"
            "order. Each concept speaks to its angle's audience or use. Pick the best format for each\n"
            f"angle; formats may repeat across angles.\nANGLES:\n{angles}\n\nFormats:"
        )
    if request.vary_format:
        fmt = "Pick the best format for each angle; formats may differ. Keep everything else as alike as you can."
        composition = ""
    else:
        if request.test_format:
            fmt = f"Use the format {request.test_format} for EVERY concept."
        else:
            fmt = (
                "Pick ONE format from the list below for EVERY concept, and explain in format_rationale\n"
                "why it suits all the angles."
            )
        composition = (
            "\nKeep the composition identical: write the shared framing, setting, lighting and style once\n"
            "in shared_composition, and include it word for word in every image prompt."
        )
    return (
        f"CREATIVE TEST: create exactly {len(request.angles)} ad concepts for the same offer, one per angle\n"
        "below, in this order. This is a test of messaging: each concept must differ ONLY in its\n"
        "messaging angle - the hook, the person shown and the caption. Keep the format, the on-image\n"
        "text approach and the composition the same across angles, so any difference in results comes\n"
        f"from the angle.\n{fmt}{composition}\n"
        "For each concept, state the angle's core message in one line in core_message.\n"
        f"ANGLES:\n{angles}\n\nFormats:"
    )


def _source_text_section(request: CreativeRequest) -> str:
    if not request.source_text:
        return ""
    return (
        "SOURCE TEXT (the brand confirmed it holds the rights; quote it word for word or not at all):\n"
        f"\"\"\"\n{request.source_text}\n\"\"\"\n"
    )


def _on_image_text_section(request: CreativeRequest) -> str:
    if request.text_choice == "none":
        return "ON-IMAGE TEXT: none. Leave on_image_text empty; the image carries no words.\n"
    limit = MAX_RENDERED_WORDS if request.text_mode == "rendered" else MAX_ON_IMAGE_WORDS
    if request.text_choice == "own":
        texts = request.own_texts()
        lines = "\n".join(f"{i}. \"{t}\"" for i, t in enumerate(texts, 1))
        source = f"The brand's own words, one per creative in order (do not change them):\n{lines}"
    else:
        source = (
            f"Write on_image_text for each creative to fit its angle and format: at most {limit} words,\n"
            "in the brand voice, no placeholders, and no claim the caption could not make."
        )
    if request.text_mode == "rendered":
        mode = (
            "Mode: RENDERED. The image model draws the text. Put the exact on_image_text in double\n"
            "quotes inside the image prompt, with where and how it appears (e.g. bold white sans-serif\n"
            "across the top third). It will be read back after generation and flagged if it differs."
        )
    else:
        mode = (
            "Mode: OVERLAY. The text is added later as an editable layer. Do NOT put it in the image\n"
            "prompt; instead compose the image with clean negative space for it and say where\n"
            "(e.g. \"clean negative space in the top third\")."
        )
    return f"ON-IMAGE TEXT:\n{source}\n{mode}\n"


def build_prompt(request: CreativeRequest, trend_notes: str) -> str:
    """Return the user message that turns a creative request into a CreativePlan.

    Must contain:
      - The campaign brief (request.brief), and every brand field: brand_name,
        voice, colors, offer, audience.
      - request.goal (awareness / sales / launch) and request.platform
        (instagram / tiktok) when given; both are optional, so say what to do
        when they're None (e.g. "no specific goal", "platform-agnostic").
      - trend_notes (from config/trend_notes.md) as background context, not as
        rules; skip the section if it's empty.
      - The task: exactly 3 concepts, each in a *different* format chosen from
        ugc, model_with_product, lifestyle, studio_hero, flat_lay,
        problem_solution, seasonal - picked to fit this brand, goal and platform.
      - Per item, what each field is for:
          format    - one of the formats above (no repeats across the 3).
          rationale - one sentence on why this format suits this brand.
          prompt    - a self-contained text-to-image prompt (subject, setting,
                      lighting, composition, style, brand colors). Square 1:1.
                      No text/logos rendered in the image; no real people.
          caption   - ad copy in the brand voice, aimed at the audience.
          alt_text  - a literal description of what the image shows (not marketing copy).
          faq       - exactly 2 Q&As a shopper in this audience would ask about the product.
      - Guardrails: no claims the brief doesn't support (prices, health, guarantees).

    Hint: an f-string with clearly labelled sections works well, e.g.
    "BRIEF:\n{request.brief}\n\nBRAND:\n- Name: {brand.brand_name}\n- Voice: ...",
    then "CONTEXT" (goal, platform, trend notes), "TASK" and "RULES". Join colors
    with ", " and handle an empty list ("no fixed palette"). A one-line
    description per format helps the model pick well (e.g. "flat_lay: product
    and props arranged top-down"). The output schema is enforced by Instructor
    through CreativePlan, so describe the *content* you want rather than the
    JSON shape.
    """
    brand = request.brand
    palette = ", ".join(brand.colors) if brand.colors else "no fixed palette"
    goal = request.goal or "no specific goal"
    platform = request.platform or "platform-agnostic"
    trends = f"\nTREND NOTES (background context, not rules):\n{trend_notes}\n" if trend_notes.strip() else ""
    strategy = (
        f"\nSTRATEGY NOTES (direction for this campaign; the RULES below still apply):\n{request.strategy_notes}\n"
        if request.strategy_notes else ""
    )

    return f"""You are a senior performance creative director planning paid social ads.

BRIEF:
{request.brief}

BRAND:
- Name: {brand.brand_name}
- Voice: {brand.voice}
- Colors: {palette}
- Offer: {brand.offer}
- Audience: {brand.audience}

CONTEXT:
- Offer type: {request.offer_type}
- Goal: {goal}
- Platform: {platform}
{trends}{strategy}
{offer_type_guidance(request)}{_product_photo_section(request)}
{_source_text_section(request)}{_on_image_text_section(request)}
ANALYZE (before choosing any image):
1. What is this product, in one plain sentence a stranger would understand?
2. What specific pain or desire does it solve for this audience, and what does
   that problem look like in real life?
3. Who is the cold buyer who has never heard of this brand, and what would they
   need to see to trust it enough to buy?

TASK:
{_task_intro(request)}
{_format_menu(request.offer_type)}

For each concept:
- format: one of the formats above{"" if request.angles else ", no repeats"}
- angle: {"the angle this concept is for, copied exactly" if request.angles else "leave empty"}
- rationale: one sentence on why this format suits this brand and buyer
- prompt: a self-contained image prompt covering subject, setting, lighting,
  composition, style and the brand colors. It MUST start with its format's
  opening phrase above, and the scene must match that format. Words like
  "flat lay", "top-down" or "bird's-eye" belong only to flat_lay.
- caption: ad copy in the brand voice, written for the audience
- alt_text: a literal description of what the image shows, not marketing copy
- faq: exactly 2 questions a shopper in this audience would ask, with answers
- needs_input: every [▢ ...] placeholder used in the caption or FAQ, copied exactly
- on_image_text: {"leave empty" if request.text_choice == "none" else "the words on the image, as set out under ON-IMAGE TEXT"}
- core_message: {f"this angle's core message in one line (at most {MAX_CORE_MESSAGE_WORDS} words)" if request.creative_test else "leave empty"}

CHECK (every concept must pass all three, or revise it):
- Clarity: would a buyer understand what is being sold within 2 seconds?
- Pain solved: can the buyer SEE their problem being solved, not just the
  product sitting there?
- Stranger test: would someone who never knew this brand buy from this image alone?

RULES:
- Square 1:1 composition.
- No logos or watermarks in the image. No words in the image, except the exact
  on_image_text when ON-IMAGE TEXT says RENDERED.
{_props_rule(request)}- No placeholders in on_image_text, ever.
- People must be fictional. Never depict a real or recognizable person.
{REALISM_RULE}- Captions must not read as a real customer testimonial (no names, no "I've
  used this for 3 weeks"). Speak as the brand.
- Classify every claim in the caption and FAQ before writing it:
  * from-brief: stated in the BRIEF or BRAND above. Use it as given.
  * general-category: true of this kind of product in general (e.g. "trail
    shoes have lugged soles"). Allowed, but never phrased as a brand promise.
  * brand-specific: a fact about THIS brand or product that is not in the
    brief - policies, returns, prices, discounts, specs, materials, sizes,
    shipping, guarantees, warranties, certifications, awards, results.
    NEVER invent these. Write a placeholder instead, in exactly this form:
    [▢ description, e.g. example]
- A placeholder replaces the WHOLE unsupported claim, not just the number or
  detail inside it. Surrounding words must not assert anything either.
    Wrong: "Free returns within [▢ return window, e.g. 30 days]." (still claims free returns)
    Right: "[▢ return policy, e.g. free 30-day returns]"
    Wrong: "Ships free in [▢ shipping time, e.g. 3-5 days]." (still claims free shipping)
    Right: "[▢ shipping offer, e.g. free shipping in 3-5 days]"
- If an FAQ answer depends on a brand-specific fact, the answer is the
  placeholder for that whole fact, not a guess wrapped around one.
- Do not hint at future products, releases, drops, collections or "coming
  soon" news unless the brief states them.
- Placeholders go only in the caption and FAQ, never in the image prompt or
  alt_text.
- Keep brand-specific facts in the FAQ. Put a placeholder in the caption only
  when that sentence cannot work without the fact; otherwise write the caption
  without it and let the FAQ carry the detail.
- Quote only from SOURCE TEXT, word for word. Never quote or paraphrase closely
  from books, lyrics, articles or other people's text that is not given there.
- No health or medical claims, and no performance results, unless the brief states them.
"""


def plan_creatives(request: CreativeRequest) -> tuple[CreativePlan, Decimal]:
    """Ask the model for 3 creatives; returns the plan and its real USD cost."""
    plan, completion = client.chat.completions.create_with_completion(
        model=PLANNING_MODEL,
        response_model=CreativePlan,
        messages=[{"role": "user", "content": build_prompt(request, load_trend_notes())}],
        context={"spec": PlanSpec.from_request(request)},
        max_tokens=2000,
        max_retries=2,
    )
    usage = completion.usage
    cost = planning_cost(usage.prompt_tokens, usage.completion_tokens)
    logger.info("Creative plan ready. Tokens: %s, cost: $%s", usage.total_tokens, cost)
    return plan, cost
