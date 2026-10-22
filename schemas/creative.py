from typing import Literal, Optional

from pydantic import BaseModel, Field


class Brand(BaseModel):
    brand_name: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    colors: list[str] = Field(default_factory=list)
    product: str = Field(min_length=1)
    audience: str = Field(min_length=1)


class CreativeRequest(BaseModel):
    brief: str = Field(min_length=1)
    brand: Brand
    goal: Optional[Literal["awareness", "sales", "launch"]] = None
    platform: Optional[Literal["instagram", "tiktok"]] = None
