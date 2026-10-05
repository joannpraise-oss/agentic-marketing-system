from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.errors import register_error_handlers
from core.log_config import configure_logging
from core.middleware import request_logging
from db.database import create_tables, engine
from routers.campaign import router

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield
    engine.dispose()


app = FastAPI(lifespan=lifespan)

# Middleware added last is outermost: CORS must wrap logging so every response gets CORS headers.
app.middleware("http")(request_logging)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://agentic-marketing-ui.vercel.app", "https://brkeven.com", "https://www.brkeven.com", "https://app.brkeven.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
register_error_handlers(app)

app.include_router(router)


@app.get("/")
def hello():
    return {"message": "Agentic Marketing Analyzer v1"}
