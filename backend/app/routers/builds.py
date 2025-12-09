from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import BuildResponse
from app.optimiseur import run_optimizer  # GA
from app.optimiseur_2 import run_optimizer_cp  # CP-SAT

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
    stats: list[str] | None = Query(default=None, alias="stats"),
    effective_mastery: list[str] | None = Query(default=None, alias="effective_mastery"),
    effective_weight: float = 10.0,
    zero_component_weights: bool = False,
    require_epic: bool = False,
    require_relic: bool = False,
    solver: str = Query("ga", pattern="^(ga|cp)$"),
    prioritize_pa: bool = False,
    prioritize_pm: bool = False,
    ban_ids: list[int] | None = Query(default=None, alias="ban_ids"),
    ban_names: list[str] | None = Query(default=None, alias="ban_names"),
    avoid_negative: list[str] | None = Query(default=None, alias="avoid_negative"),
):
    if solver == "cp":
        best_build = run_optimizer_cp(
            db,
            level,
            top_k=top_k,
            force_legendary=force_legendary,
            target_stats=stats,
            effective_mastery=effective_mastery,
            effective_weight=effective_weight,
            zero_component_weights=zero_component_weights,
            require_epic=require_epic,
            require_relic=require_relic,
            verbose=verbose,
            prioritize_pa=prioritize_pa,
            prioritize_pm=prioritize_pm,
            ban_ids=ban_ids,
            ban_names=ban_names,
            avoid_negative=avoid_negative,
        )
    else:
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
            target_stats=stats,
            effective_mastery=effective_mastery,
            effective_weight=effective_weight,
            zero_component_weights=zero_component_weights,
            require_epic=require_epic,
            require_relic=require_relic,
            prioritize_pa=prioritize_pa,
            prioritize_pm=prioritize_pm,
            ban_ids=ban_ids,
            ban_names=ban_names,
            avoid_negative=avoid_negative,
        )
    alternatives = [
        BuildResponse(score=alt[2], items=alt[0], stats=alt[1]) for alt in best_build[3]
    ] if len(best_build) > 3 else None
    return BuildResponse(
        score=best_build[2],
        items=best_build[0],
        stats=best_build[1],  # ta fonction pour agréger les stats finales
        alternatives=alternatives,
    )
