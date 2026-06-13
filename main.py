from fastapi import FastAPI
from db.database import create_tables
from routers.campaign import router

app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_tables()

app.include_router(router)

@app.get("/")
def hello():
    return {"message": "Agentic Marketing Analyzer v1"}
