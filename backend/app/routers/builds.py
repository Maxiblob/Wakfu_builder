from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import BuildResponse
from app.optimiseur import run_optimizer  # ta fonction corrigée

router = APIRouter()

@router.get("/optimise/{level}", response_model=BuildResponse)
def get_best_build(db: Session = Depends(get_db), level: int = 245):
    best_build = run_optimizer(db, level)  # ton algo génétique
    return BuildResponse(
        score=best_build[2],
        items=best_build[0],
        stats=best_build[1]  # ta fonction pour agréger les stats finales
    )