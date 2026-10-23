from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Brand(BaseModel):
    brand_name: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    colors: list[str] = Field(default_factory=list)
    offer: str = Field(min_length=1)  # the product, service, content or idea being promoted
    audience: str = Field(min_length=1)


MAX_ANGLES = 5
MAX_ANGLE_CHARS = 60

OfferType = Literal["physical_product", "service", "content", "idea_test"]


class CreativeRequest(BaseModel):
    brief: str = Field(min_length=1)
    brand: Brand
    offer_type: OfferType = "physical_product"
    # One creative per angle, all for the same offer. Without angles the run makes 3 creatives.
    angles: Optional[list[str]] = Field(default=None, min_length=1, max_length=MAX_ANGLES)

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
    goal: Optional[Literal["awareness", "sales", "launch"]] = None
    platform: Optional[Literal["instagram", "tiktok"]] = None
