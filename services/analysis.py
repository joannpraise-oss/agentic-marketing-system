import instructor
from langsmith import traceable
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)

from langsmith import wrappers

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
wrapped_client = wrappers.wrap_openai(openai_client)
client = instructor.from_openai(wrapped_client)
class AnalysisOutput(BaseModel):
    performance_diagnosis: str
    root_cause: str
    optimization_suggestion: str
    confidence_score: float
    metrics_used: list[str]
    missing_metrics: list[str]

def calculate_metrics(data: dict) -> dict:
    enriched = dict(data)
    if data.get("spend") and data.get("revenue"):
        enriched["roas"] = round(data["revenue"] / data["spend"], 2)
    if data.get("clicks") and data.get("impressions"):
        enriched["ctr"] = round((data["clicks"] / data["impressions"]) * 100, 2)
    if data.get("spend") and data.get("clicks"):
        enriched["cpc"] = round(data["spend"] / data["clicks"], 2)
    if data.get("clicks") and data.get("conversions"):
        enriched["conversion_rate"] = round((data["conversions"] / data["clicks"]) * 100, 2)
    return enriched

@traceable
def analyze_campaign(data: dict) -> tuple[AnalysisOutput, int]:
    try:
        enriched = calculate_metrics(data)

        product_context = ""
        if enriched.get("product_price"):
            product_context += f"Product price: ${enriched['product_price']}. "
        if enriched.get("traffic_source"):
            product_context += f"Traffic source: {enriched['traffic_source']}. "

        prompt = f"""
You are an expert marketing performance analyst helping both sophisticated marketers and solo founders understand their ad campaigns.

Here is the campaign data provided:
{enriched}

{product_context}

Analyze this campaign and provide:
1. A clear performance diagnosis in plain English
2. The most likely root cause of any issues
3. A single specific optimization suggestion they can act on today
4. A confidence score from 0.0 to 1.0 based on how complete the data is
5. List which metrics you used
6. List any important metrics that are missing and would improve the analysis

Important: Give ONE specific action, not a menu of options.
If data is limited, still provide the best analysis you can and be honest about confidence level.
Keep language simple enough for a first-time advertiser to understand.
"""

        response = client.chat.completions.create_with_completion(
            model="gpt-4o-mini",
            response_model=AnalysisOutput,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        result, completion = response
        tokens_used = completion.usage.total_tokens
        logger.info(f"Analysis complete. Tokens used: {tokens_used}")
        return result, tokens_used

    except Exception as e:
        logger.error(f"LLM analysis failed: {str(e)}")
        raise
