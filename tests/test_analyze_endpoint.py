from sqlmodel import Session, select

from db.models import AnalysisResult, Campaign

PAYLOAD = {"spend": 500, "revenue": 2000, "product_price": 32, "traffic_source": "Meta Ads"}


def test_happy_path(client, engine, mock_openai):
    resp = client.post("/analyze_campaign", json=PAYLOAD)

    assert resp.status_code == 200
    body = resp.json()
    assert body["diagnosis"] == "Healthy campaign."
    assert body["confidence_score"] == 0.8
    assert body["tokens_used"] == 321
    assert body["metrics_used"] == ["spend", "revenue"]

    with Session(engine) as session:
        campaign = session.exec(select(Campaign)).one()
        result = session.exec(select(AnalysisResult)).one()
    assert campaign.id == body["campaign_id"]
    assert campaign.traffic_source == "Meta Ads"
    assert result.campaign_id == campaign.id
    assert result.tokens_used == 321


def test_openai_called_once_with_model_and_enriched_prompt(client, mock_openai):
    client.post("/analyze_campaign", json=PAYLOAD)

    (call,) = mock_openai.calls
    assert call["model"] == "gpt-4o-mini"
    assert "'roas': 4.0" in call["messages"][0]["content"]


def test_missing_required_field_is_400_in_standard_shape(client, mock_openai):
    resp = client.post("/analyze_campaign", json={"spend": 100})

    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "invalid_input"
    assert body["request_id"]
    assert body["errors"] == [{"field": "revenue", "message": "Field required"}]
    assert mock_openai.calls == []


def test_unknown_route_uses_standard_shape(client):
    resp = client.get("/nope")

    assert resp.status_code == 404
    assert resp.json()["code"] == "http_404"
