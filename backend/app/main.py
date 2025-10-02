from fastapi import FastAPI
from app.routers import builds, items
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Wakfu Build Optimizer",
    description="API pour générer et analyser des builds automatiquement",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ton frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Brancher les routes
app.include_router(builds.router, prefix="/builds", tags=["builds"])
app.include_router(items.router, prefix="/items", tags=["items"])