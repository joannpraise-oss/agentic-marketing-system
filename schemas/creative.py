from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from core.placeholders import find_placeholders


class Brand(BaseModel):
    brand_name: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    colors: list[str] = Field(default_factory=list)
    offer: str = Field(min_length=1)  # the product, service, content or idea being promoted
    audience: str = Field(min_length=1)


DEFAULT_CREATIVE_COUNT = 3
MAX_ANGLES = 5
MIN_TEST_ANGLES = 2
MAX_ANGLE_CHARS = 60
MAX_SOURCE_TEXT_CHARS = 2000
MAX_STRATEGY_NOTES_CHARS = 1500
MAX_ON_IMAGE_WORDS = 12
MAX_RENDERED_WORDS = 5  # image models only spell short text reliably

OfferType = Literal["physical_product", "service", "content"]

CreativeFormat = Literal[
    "ugc", "model_with_product", "lifestyle", "studio_hero", "flat_lay", "problem_solution", "seasonal",
    "before_after", "experience", "mood_scene", "narrator",
]

# Visuals must fit the offer: products show the product, services the result and experience,
# content a mood or a narrator.
FORMATS_BY_OFFER_TYPE: dict[str, tuple[str, ...]] = {
    "physical_product": (
        "ugc", "model_with_product", "lifestyle", "studio_hero", "flat_lay", "problem_solution", "seasonal",
    ),
    "service": ("before_after", "experience", "problem_solution", "ugc", "lifestyle", "seasonal"),
    "content": ("mood_scene", "narrator", "lifestyle", "seasonal"),
}

TextChoice = Literal["own", "suggest", "none"]
TextMode = Literal["overlay", "rendered"]


def on_image_text_problem(text: str, text_mode: str) -> Optional[str]:
    """Why this on-image text is not allowed, or None if it is."""
    limit = MAX_RENDERED_WORDS if text_mode == "rendered" else MAX_ON_IMAGE_WORDS
    if not text.strip():
        return "on_image_text must not be empty."
    if len(text.split()) > limit:
        return f"on_image_text must be at most {limit} words in {text_mode} mode: {text!r}."
    if find_placeholders([text]):
        return f"on_image_text must not contain [▢ ...] placeholders: {text!r}."
    return None


class CreativeRequest(BaseModel):
    brief: str = Field(min_length=1)
    brand: Brand
    offer_type: OfferType = "physical_product"
    goal: Optional[Literal["awareness", "sales", "launch"]] = None
    platform: Optional[Literal["instagram", "tiktok"]] = None
    # One creative per angle, all for the same offer. Without angles the run makes 3 creatives.
    angles: Optional[list[str]] = Field(default=None, min_length=1, max_length=MAX_ANGLES)
    # A creative test: one creative per angle where ONLY the messaging angle varies, grouped
    # by test_group_id so results can be compared. Needs 2-5 angles.
    creative_test: bool = False
    # The one format every angle uses; if missing, the planner picks one and explains why.
    test_format: Optional[CreativeFormat] = None
    # Let each angle use a different format (test_format must then be empty).
    vary_format: bool = False
    # A snippet to quote (e.g. from a book or post); the caller must confirm they hold the rights.
    source_text: Optional[str] = Field(default=None, max_length=MAX_SOURCE_TEXT_CHARS)
    has_text_rights: bool = False
    # Campaign direction for the planner; the Strategize feature will fill this.
    strategy_notes: Optional[str] = Field(default=None, max_length=MAX_STRATEGY_NOTES_CHARS)
    # Text on the image: the caller's own words, the planner's suggestion, or none.
    text_choice: TextChoice = "none"
    # For "own": one line for every creative, or a list with one line per creative.
    on_image_text: Optional[Union[str, list[str]]] = None
    # overlay: the image leaves clean negative space and the text is added later (editable, free).
    # rendered: the model draws the text into the image (short text only; checked after generation).
    text_mode: TextMode = "overlay"

    @field_validator("angles")
    @classmethod
    def angles_are_clean_and_distinct(cls, angles: Optional[list[str]]) -> Optional[list[str]]:
        if angles is None:
            return None
        cleaned = [a.strip() for a in angles]
        if any(not a or len(a) > MAX_ANGLE_CHARS for a in cleaned):
            raise ValueError(f"Each angle must be 1-{MAX_ANGLE_CHARS} characters.")
        if len({a.casefold() for a in cleaned}) != len(cleaned):
            raise ValueError("Angles must be distinct.")
        return cleaned

    @field_validator("strategy_notes")
    @classmethod
    def blank_strategy_notes_are_none(cls, notes: Optional[str]) -> Optional[str]:
        if notes is None:
            return None
        return notes.strip() or None

    @model_validator(mode="after")
    def creative_test_settings(self) -> "CreativeRequest":
        if not self.creative_test:
            if self.test_format is not None or self.vary_format:
                raise ValueError("test_format and vary_format are only used with creative_test: true.")
            return self
        if not self.angles or len(self.angles) < MIN_TEST_ANGLES:
            raise ValueError(f"A creative test needs {MIN_TEST_ANGLES}-{MAX_ANGLES} angles.")
        if self.test_format is not None:
            if self.vary_format:
                raise ValueError("Use test_format (one format for all angles) or vary_format, not both.")
            if self.test_format not in FORMATS_BY_OFFER_TYPE[self.offer_type]:
                allowed = ", ".join(FORMATS_BY_OFFER_TYPE[self.offer_type])
                raise ValueError(f"test_format {self.test_format} does not fit a {self.offer_type} offer. Use one of: {allowed}.")
        return self

    @model_validator(mode="after")
    def source_text_needs_rights(self) -> "CreativeRequest":
        if self.source_text is not None and not self.source_text.strip():
            self.source_text = None
        if self.source_text and not self.has_text_rights:
            raise ValueError("source_text needs has_text_rights: true (confirm you have the rights to this text).")
        return self

    @model_validator(mode="after")
    def on_image_text_matches_choice(self) -> "CreativeRequest":
        if self.text_choice != "own":
            if self.on_image_text is not None:
                raise ValueError("on_image_text is only sent with text_choice 'own'.")
            return self
        if self.on_image_text is None:
            raise ValueError("text_choice 'own' needs on_image_text: one line, or one per creative.")
        if isinstance(self.on_image_text, list) and len(self.on_image_text) != self.creative_count():
            raise ValueError(
                f"on_image_text has {len(self.on_image_text)} lines but the run makes {self.creative_count()} creatives."
            )
        for text in self.own_texts():
            problem = on_image_text_problem(text, self.text_mode)
            if problem:
                raise ValueError(problem)
        return self

    def creative_count(self) -> int:
        return len(self.angles) if self.angles else DEFAULT_CREATIVE_COUNT

    def own_texts(self) -> Optional[list[str]]:
        """The caller's on-image text, one line per creative, or None."""
        if self.text_choice != "own" or self.on_image_text is None:
            return None
        if isinstance(self.on_image_text, str):
            return [self.on_image_text.strip()] * self.creative_count()
        return [t.strip() for t in self.on_image_text]
