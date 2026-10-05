from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.database import create_tables, engine
from routers.campaign import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield
    engine.dispose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://agentic-marketing-ui.vercel.app", "https://brkeven.com", "https://www.brkeven.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def hello():
    return {"message": "Agentic Marketing Analyzer v1"}
