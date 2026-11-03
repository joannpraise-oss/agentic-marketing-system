from decimal import Decimal
from sqlalchemy import JSON, Column, Numeric, event
from sqlmodel import SQLModel, Field
from typing import Any, Optional
from datetime import datetime, timezone

from core.placeholders import PlaceholdersRemainError, needs_input


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
    offer_type: Optional[str] = Field(default=None)  # physical_product | service | content
    angle: Optional[str] = Field(default=None)  # the requested angle this creative was made for
    text_rights_confirmed: Optional[bool] = Field(default=None)  # set when the request quoted source_text
    text_choice: Optional[str] = Field(default=None)  # own | suggest | none
    text_mode: Optional[str] = Field(default=None)  # overlay | rendered (None when there is no text)
    on_image_text: Optional[str] = Field(default=None)
    # Creative tests: every asset of one test shares test_group_id; core_message is its angle's message.
    test_group_id: Optional[str] = Field(default=None, index=True)
    # fixed_format: only the angle varies, so results compare by angle.
    # varied_format: angle and format both vary, so results must be reported as angle + format.
    test_mode: Optional[str] = Field(default=None)
    core_message: Optional[str] = Field(default=None)
    # Product-photo runs: did the generated label show brand_name? None = not checked (no photo).
    label_ok: Optional[bool] = Field(default=None)
    label_read: Optional[str] = Field(default=None)  # the label text as read; None if the check failed
    status: str = Field(default="draft")  # "draft" | "approved" | "archived"
    cost_usd: Decimal = Field(default=Decimal("0"), sa_column=Column(Numeric(10, 4), nullable=False))
    created_at: datetime = Field(default_factory=utcnow)


@event.listens_for(Asset, "before_insert")
@event.listens_for(Asset, "before_update")
def _block_approval_while_placeholders_remain(mapper, connection, asset: Asset) -> None:
    # Runs on every flush, so no code path can save an approved asset with unfilled placeholders
    # or a label that failed its check.
    if asset.status == "approved":
        remaining = needs_input(asset.caption, asset.alt_text, asset.faq, asset.label_ok)
        if remaining:
            raise PlaceholdersRemainError(remaining)
