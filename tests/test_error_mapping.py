import httpx
import openai
import pytest
from instructor.core.exceptions import InstructorRetryException
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, select

from db.database import get_session
from db.models import AnalysisResult, Campaign
from main import app

PAYLOAD = {"spend": 500, "revenue": 2000}
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def status_error(cls, status):
    return cls("upstream said no", response=httpx.Response(status, request=REQUEST), body=None)


# The rate-limit / overload (503 + Retry-After) mapping is intentionally not covered here.
CASES = [
    pytest.param(lambda: status_error(openai.AuthenticationError, 401), 500, "internal_error", id="auth"),
    pytest.param(lambda: status_error(openai.PermissionDeniedError, 403), 500, "internal_error", id="permission"),
    pytest.param(lambda: openai.APITimeoutError(request=REQUEST), 504, "ai_timeout", id="timeout"),
    pytest.param(lambda: openai.APIConnectionError(request=REQUEST), 502, "ai_unavailable", id="connection"),
    pytest.param(lambda: status_error(openai.BadRequestError, 400), 502, "ai_bad_response", id="bad-request"),
    pytest.param(lambda: InstructorRetryException("validation failed", n_attempts=3, total_usage=0), 502, "ai_bad_response", id="instructor-retry"),
    pytest.param(lambda: RuntimeError("boom"), 500, "internal_error", id="unexpected"),
]


@pytest.mark.parametrize("make_error, status, code", CASES)
def test_openai_failure_maps_to_status_and_keeps_campaign(client, engine, mock_openai, make_error, status, code):
    mock_openai.error = make_error()

    resp = client.post("/analyze_campaign", json=PAYLOAD)

    assert resp.status_code == status
    body = resp.json()
    assert body["code"] == code
    assert body["request_id"] == resp.headers["X-Request-ID"]
    with Session(engine) as session:
        campaign = session.exec(select(Campaign)).one()
        assert session.exec(select(AnalysisResult)).all() == []
    assert body["campaign_id"] == campaign.id


def test_auth_failure_does_not_leak_details(client, mock_openai):
    mock_openai.error = status_error(openai.AuthenticationError, 401)

    resp = client.post("/analyze_campaign", json=PAYLOAD)

    text = resp.text.lower()
    assert "openai" not in text and "api key" not in text and "upstream said no" not in text


def test_database_error_on_save_is_db_error(client, engine, mock_openai):
    def broken_session():
        with Session(engine) as session:
            def fail():
                raise OperationalError("INSERT", {}, Exception("connection lost"))
            session.commit = fail
            yield session

    app.dependency_overrides[get_session] = broken_session

    resp = client.post("/analyze_campaign", json=PAYLOAD)

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "db_error"
    assert "connection lost" not in resp.text
    assert "campaign_id" not in body  # nothing was saved
    assert mock_openai.calls == []


def test_unhandled_500_still_carries_cors_headers(client, mock_openai):
    mock_openai.error = RuntimeError("boom")

    resp = client.post("/analyze_campaign", json=PAYLOAD, headers={"Origin": "https://brkeven.com"})

    assert resp.status_code == 500
    assert resp.headers["access-control-allow-origin"] == "https://brkeven.com"
