from pydantic import BaseModel
from typing import Optional

class CampaignInput(BaseModel):
    spend: float
    revenue: float
    clicks: Optional[int] = None
    impressions: Optional[int] = None
    conversions: Optional[int] = None
    ctr: Optional[float] = None
    cpc: Optional[float] = None
    roas: Optional[float] = None
    conversion_rate: Optional[float] = None
    product_price: Optional[float] = None
    traffic_source: Optional[str] = None
