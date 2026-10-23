from typing import Literal, Optional

from pydantic import BaseModel, Field


class Brand(BaseModel):
    brand_name: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    colors: list[str] = Field(default_factory=list)
    offer: str = Field(min_length=1)  # the product, service, content or idea being promoted
    audience: str = Field(min_length=1)


OfferType = Literal["physical_product", "service", "content", "idea_test"]


class CreativeRequest(BaseModel):
    brief: str = Field(min_length=1)
    brand: Brand
    offer_type: OfferType = "physical_product"
    goal: Optional[Literal["awareness", "sales", "launch"]] = None
    platform: Optional[Literal["instagram", "tiktok"]] = None
