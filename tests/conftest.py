import os
from types import SimpleNamespace

# Must run before any app module is imported: no real DB, no real OpenAI, no tracing, no .env.
import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: False
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["OPENAI_API_KEY"] = "sk-test-not-real"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import services.analysis as analysis_module
from db.database import get_session
from main import app


@pytest.fixture
def engine():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def client(engine):
    def override_get_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    # raise_server_exceptions=False: unhandled errors become real 500 responses, as in production.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_analysis(**overrides):
    fields = dict(
        performance_diagnosis="Healthy campaign.",
        root_cause="Strong targeting.",
        optimization_suggestion="Raise budget 20%.",
        confidence_score=0.8,
        metrics_used=["spend", "revenue"],
        missing_metrics=["ctr"],
    )
    fields.update(overrides)
    return analysis_module.AnalysisOutput(**fields)


@pytest.fixture
def mock_openai(monkeypatch):
    """Replace the instructor client; returns a handle to set a result or an exception."""
    state = SimpleNamespace(result=make_analysis(), tokens=321, error=None, calls=[])

    def create_with_completion(**kwargs):
        state.calls.append(kwargs)
        if state.error is not None:
            raise state.error
        return state.result, SimpleNamespace(usage=SimpleNamespace(total_tokens=state.tokens))

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create_with_completion=create_with_completion))
    )
    monkeypatch.setattr(analysis_module, "client", fake_client)
    return state
