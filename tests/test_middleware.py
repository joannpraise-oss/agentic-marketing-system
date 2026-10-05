import logging


def test_generates_and_returns_request_id(client):
    resp = client.get("/")

    assert resp.status_code == 200
    assert len(resp.headers["X-Request-ID"]) == 12


def test_honours_incoming_request_id(client):
    resp = client.get("/", headers={"X-Request-ID": "abc-123"})

    assert resp.headers["X-Request-ID"] == "abc-123"


def test_logs_one_line_per_request(client, caplog):
    with caplog.at_level(logging.INFO, logger="core.middleware"):
        client.get("/")

    (record,) = [r for r in caplog.records if r.name == "core.middleware"]
    assert "GET / -> 200" in record.getMessage()


def test_failed_request_is_still_logged_with_traceback(client, mock_openai, caplog):
    mock_openai.error = RuntimeError("boom")

    with caplog.at_level(logging.INFO, logger="core.middleware"):
        client.post("/analyze_campaign", json={"spend": 1, "revenue": 1})

    messages = [r.getMessage() for r in caplog.records if r.name == "core.middleware"]
    assert any("Unhandled error" in m for m in messages)
    assert any("-> 500" in m for m in messages)
    assert any(r.exc_info for r in caplog.records if r.name == "core.middleware")
