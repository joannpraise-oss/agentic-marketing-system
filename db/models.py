from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone


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
