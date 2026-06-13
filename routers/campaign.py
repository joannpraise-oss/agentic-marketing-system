from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from db.models import Campaign, AnalysisResult
from db.database import get_session
from services.analysis import analyze_campaign
from schemas.campaign import CampaignInput
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze_campaign")
async def analyze_campaign_endpoint(data: CampaignInput, session: Session = Depends(get_session)):
    try:
        campaign = Campaign(
            spend=data.spend,
            ctr=data.ctr or 0.0,
            cpc=data.cpc or 0.0,
            roas=data.roas or 0.0,
            conversion_rate=data.conversion_rate or 0.0,
            product_price=data.product_price,
            traffic_source=data.traffic_source  
        )
        session.add(campaign)
        session.commit()
        session.refresh(campaign)
        logger.info(f"Campaign saved with id: {campaign.id}")

    except Exception as e:
        logger.error(f"Database error saving campaign: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save campaign data. Please try again.")

    try:
        input_data = data.dict(exclude_none=True)
        analysis, tokens_used = analyze_campaign(input_data)

        result = AnalysisResult(
            campaign_id=campaign.id,
            diagnosis=analysis.performance_diagnosis,
            root_cause=analysis.root_cause,
            suggestion=analysis.optimization_suggestion,
            confidence=analysis.confidence_score,
            tokens_used=tokens_used
        )
        session.add(result)
        session.commit()
        logger.info(f"Analysis saved for campaign {campaign.id}. Tokens: {tokens_used}")

    except Exception as e:
        logger.error(f"Analysis failed for campaign {campaign.id}: {str(e)}")
        raise HTTPException(status_code=503, detail="Analysis service temporarily unavailable. Campaign data was saved.")

    return {
        "campaign_id": campaign.id,
        "diagnosis": analysis.performance_diagnosis,
        "root_cause": analysis.root_cause,
        "optimization_suggestion": analysis.optimization_suggestion,
        "confidence_score": analysis.confidence_score,
        "metrics_used": analysis.metrics_used,
        "missing_metrics": analysis.missing_metrics,
        "tokens_used": tokens_used
    }
