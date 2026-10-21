import logging
import os
from decimal import Decimal

import instructor
from dotenv import load_dotenv
from langsmith import wrappers
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from schemas.creative import Brand

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


class FAQ(BaseModel):
    question: str
    answer: str


class CreativeItem(BaseModel):
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
        return items


def planning_cost(prompt_tokens: int, completion_tokens: int) -> Decimal:
    return (
        Decimal(prompt_tokens) * INPUT_USD_PER_MTOK + Decimal(completion_tokens) * OUTPUT_USD_PER_MTOK
    ) / Decimal(1_000_000)


def build_prompt(brief: str, brand: Brand) -> str:
    """Return the user message that turns a brief + brand into a CreativePlan.

    Must contain:
      - The campaign brief, and every brand field: brand_name, voice, colors,
        product, audience.
      - The task: exactly 3 *distinct* concepts (different scene, angle or hook,
        not three rewordings of one idea).
      - Per item, what each field is for:
          prompt   - a self-contained text-to-image prompt (subject, setting,
                     lighting, composition, style, brand colors). Square 1:1.
                     No text/logos rendered in the image; no real people.
          caption  - ad copy in the brand voice, aimed at the audience.
          alt_text - a literal description of what the image shows (not marketing copy).
          faq      - exactly 2 Q&As a shopper in this audience would ask about the product.
      - Guardrails: no claims the brief doesn't support (prices, health, guarantees).

    Hint: an f-string with clearly labelled sections works well, e.g.
    "BRIEF:\n{brief}\n\nBRAND:\n- Name: {brand.brand_name}\n- Voice: ...", then
    a "TASK" and "RULES" section. Join colors with ", " and handle an empty list
    ("no fixed palette"). The output schema is enforced by Instructor through
    CreativePlan, so describe the *content* you want rather than the JSON shape.
    """
    raise NotImplementedError("build_prompt is yours to write")


def plan_creatives(brief: str, brand: Brand) -> tuple[CreativePlan, Decimal]:
    """Ask the model for 3 creatives; returns the plan and its real USD cost."""
    plan, completion = client.chat.completions.create_with_completion(
        model=PLANNING_MODEL,
        response_model=CreativePlan,
        messages=[{"role": "user", "content": build_prompt(brief, brand)}],
        max_tokens=2000,
        max_retries=2,
    )
    usage = completion.usage
    cost = planning_cost(usage.prompt_tokens, usage.completion_tokens)
    logger.info("Creative plan ready. Tokens: %s, cost: $%s", usage.total_tokens, cost)
    return plan, cost
