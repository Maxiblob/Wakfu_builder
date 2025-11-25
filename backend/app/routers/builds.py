from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import BuildResponse
from app.optimiseur import run_optimizer  # ta fonction corrigée

router = APIRouter()

@router.get("/optimise/{level}", response_model=BuildResponse)
def get_best_build(
    db: Session = Depends(get_db),
    level: int = 245,
    top_k: int = 25,
    pop_size: int = 60,
    generations: int = 80,
    elite: int = 3,
    proba_mutation: float = 0.2,
    verbose: bool = False,
    force_legendary: bool = False,
):
    best_build = run_optimizer(
        db,
        level,
        top_k=top_k,
        pop_size=pop_size,
        generations_max=generations,
        elite=elite,
        proba_mutation=proba_mutation,
        verbose=verbose,
        force_legendary=force_legendary,
    )
    return BuildResponse(
        score=best_build[2],
        items=best_build[0],
        stats=best_build[1]  # ta fonction pour agréger les stats finales
    )
