import logging

from fastapi import APIRouter, Depends, Request
from sqlmodel import Session

from db.database import get_session
from db.models import AnalysisResult, Campaign
from schemas.campaign import CampaignInput
from services.analysis import analyze_campaign

router = APIRouter()
logger = logging.getLogger(__name__)


# Plain `def` (not async): the DB and OpenAI calls are blocking, so FastAPI runs this in its threadpool.
@router.post("/analyze_campaign")
def analyze_campaign_endpoint(data: CampaignInput, request: Request, session: Session = Depends(get_session)):
    campaign = Campaign(
        spend=data.spend,
        ctr=data.ctr or 0.0,
        cpc=data.cpc or 0.0,
        roas=data.roas or 0.0,
        conversion_rate=data.conversion_rate or 0.0,
        product_price=data.product_price,
        traffic_source=data.traffic_source,
    )
    session.add(campaign)
    session.commit()
    session.refresh(campaign)
    logger.info(f"Campaign saved with id: {campaign.id}")

    # From here on the campaign is saved, so any failure response reports its id.
    # The exception itself is mapped to a status code by core.errors.
    request.state.campaign_id = campaign.id

    analysis, tokens_used = analyze_campaign(data.model_dump(exclude_none=True))

    result = AnalysisResult(
        campaign_id=campaign.id,
        diagnosis=analysis.performance_diagnosis,
        root_cause=analysis.root_cause,
        suggestion=analysis.optimization_suggestion,
        confidence=analysis.confidence_score,
        tokens_used=tokens_used,
    )
    session.add(result)
    session.commit()
    logger.info(f"Analysis saved for campaign {campaign.id}. Tokens: {tokens_used}")

    return {
        "campaign_id": campaign.id,
        "diagnosis": analysis.performance_diagnosis,
        "root_cause": analysis.root_cause,
        "optimization_suggestion": analysis.optimization_suggestion,
        "confidence_score": analysis.confidence_score,
        "metrics_used": analysis.metrics_used,
        "missing_metrics": analysis.missing_metrics,
        "tokens_used": tokens_used,
    }
