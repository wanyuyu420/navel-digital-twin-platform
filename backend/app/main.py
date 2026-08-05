from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router

app = FastAPI(title="Gannan Navel Orange Digital Twin", version="1.0.0")

app.include_router(api_router, prefix="/api/v1")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Gannan Navel Orange Digital Twin API", "status": "ok"}


@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "service": "fastapi"}
