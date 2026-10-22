import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Literal

import instructor
from dotenv import load_dotenv
from langsmith import wrappers
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from schemas.creative import CreativeRequest

load_dotenv()

logger = logging.getLogger(__name__)

client = instructor.from_openai(wrappers.wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY"))))

PLANNING_MODEL = "gpt-4o-mini"
# gpt-4o-mini list price, USD per 1M tokens. Update if the model or price changes.
INPUT_USD_PER_MTOK = Decimal("0.15")
OUTPUT_USD_PER_MTOK = Decimal("0.60")
# Planning cost is only known afterwards; this reserve goes into the pre-run spend check.
PLANNING_RESERVE_USD = Decimal("0.01")
NUM_CREATIVES = 3
TREND_NOTES_PATH = Path(__file__).resolve().parent.parent / "config" / "trend_notes.md"

CreativeFormat = Literal[
    "ugc", "model_with_product", "lifestyle", "studio_hero", "flat_lay", "problem_solution", "seasonal"
]


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


class CreativePlan(BaseModel):
    items: list[CreativeItem] = Field(min_length=NUM_CREATIVES, max_length=NUM_CREATIVES)

    @field_validator("items")
    @classmethod
    def prompts_are_distinct(cls, items: list[CreativeItem]) -> list[CreativeItem]:
        # A validation error here makes Instructor re-ask the model.
        prompts = {item.prompt.strip().lower() for item in items}
        if len(prompts) != len(items):
            raise ValueError("Each creative needs a distinct image prompt.")
        if len({item.format for item in items}) != len(items):
            raise ValueError("Each creative must use a different format.")
        return items


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


def build_prompt(request: CreativeRequest, trend_notes: str) -> str:
    """Return the user message that turns a creative request into a CreativePlan.

    Must contain:
      - The campaign brief (request.brief), and every brand field: brand_name,
        voice, colors, product, audience.
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

    return f"""You are a senior performance creative director planning paid social ads.

BRIEF:
{request.brief}

BRAND:
- Name: {brand.brand_name}
- Voice: {brand.voice}
- Colors: {palette}
- Product: {brand.product}
- Audience: {brand.audience}

CONTEXT:
- Goal: {goal}
- Platform: {platform}
{trends}
ANALYZE (before choosing any image):
1. What is this product, in one plain sentence a stranger would understand?
2. What specific pain or desire does it solve for this audience, and what does
   that problem look like in real life?
3. Who is the cold buyer who has never heard of this brand, and what would they
   need to see to trust it enough to buy?

TASK:
Create exactly 3 ad concepts, each in a DIFFERENT format chosen to fit this
brand, goal and platform:
- ugc: phone-camera feel, a fictional person using the product at home
- model_with_product: a fictional model wearing, holding or applying it
- lifestyle: the product in a styled real-world setting
- studio_hero: clean background, the product as the star
- flat_lay: product and props arranged top-down
- problem_solution: shows the pain point and the product solving it
- seasonal: tied to a holiday, gifting or a timely moment

For each concept:
- format: one of the formats above, no repeats
- rationale: one sentence on why this format suits this brand and buyer
- prompt: a self-contained image prompt covering subject, setting, lighting,
  composition, style and the brand colors
- caption: ad copy in the brand voice, written for the audience
- alt_text: a literal description of what the image shows, not marketing copy
- faq: exactly 2 questions a shopper in this audience would ask, with answers

CHECK (every concept must pass all three, or revise it):
- Clarity: would a buyer understand what is being sold within 2 seconds?
- Pain solved: can the buyer SEE their problem being solved, not just the
  product sitting there?
- Stranger test: would someone who never knew this brand buy from this image alone?

RULES:
- Square 1:1 composition.
- No text, words, logos or watermarks rendered inside the image.
- People must be fictional. Never depict a real or recognizable person.
- Captions must not read as a real customer testimonial (no names, no "I've
  used this for 3 weeks"). Speak as the brand.
- Make no claims the brief does not support: no prices, health or medical
  claims, guarantees, or results.
"""


def plan_creatives(request: CreativeRequest) -> tuple[CreativePlan, Decimal]:
    """Ask the model for 3 creatives; returns the plan and its real USD cost."""
    plan, completion = client.chat.completions.create_with_completion(
        model=PLANNING_MODEL,
        response_model=CreativePlan,
        messages=[{"role": "user", "content": build_prompt(request, load_trend_notes())}],
        max_tokens=2000,
        max_retries=2,
    )
    usage = completion.usage
    cost = planning_cost(usage.prompt_tokens, usage.completion_tokens)
    logger.info("Creative plan ready. Tokens: %s, cost: $%s", usage.total_tokens, cost)
    return plan, cost
