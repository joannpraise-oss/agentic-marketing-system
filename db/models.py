from decimal import Decimal
from sqlalchemy import JSON, Column, Numeric, event
from sqlmodel import SQLModel, Field
from typing import Any, Optional
from datetime import datetime, timezone

from core.placeholders import PlaceholdersRemainError, asset_texts, find_placeholders


def utcnow() -> datetime:
    # Naive UTC, matching the existing timestamp-without-tz columns (datetime.utcnow is deprecated).
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Campaign(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ctr: float
    cpc: float
    roas: float
    conversion_rate: float
    spend: float
    product_price: Optional[float] = Field(default=None)
    traffic_source: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)

class AnalysisResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    campaign_id: int = Field(foreign_key="campaign.id")
    diagnosis: str
    root_cause: str
    suggestion: str
    confidence: float
    tokens_used: int = Field(default=0)
    created_at: datetime = Field(default_factory=utcnow)


class Asset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    campaign_brief: str
    kind: str  # "image" | "video"
    prompt: str
    model: str
    provider: str
    file_url: Optional[str] = Field(default=None)
    caption: Optional[str] = Field(default=None)
    alt_text: Optional[str] = Field(default=None)
    faq: Optional[Any] = Field(default=None, sa_column=Column(JSON))
    format: Optional[str] = Field(default=None)  # creative format, e.g. "ugc", "flat_lay"
    status: str = Field(default="draft")  # "draft" | "approved" | "archived"
    cost_usd: Decimal = Field(default=Decimal("0"), sa_column=Column(Numeric(10, 4), nullable=False))
    created_at: datetime = Field(default_factory=utcnow)


@event.listens_for(Asset, "before_insert")
@event.listens_for(Asset, "before_update")
def _block_approval_while_placeholders_remain(mapper, connection, asset: Asset) -> None:
    # Runs on every flush, so no code path can save an approved asset with unfilled placeholders.
    if asset.status == "approved":
        remaining = find_placeholders(asset_texts(asset.caption, asset.alt_text, asset.faq))
        if remaining:
            raise PlaceholdersRemainError(remaining)
