from __future__ import annotations

from typing import Iterable

import pandas as pd
from ortools.sat.python import cp_model
from sqlalchemy.orm import Session

from app.optimiseur import (
    META_COLS,
    POIDS_STATS,
    LEGENDARY_VALUES,
    REQUIRED_SLOTS,
    RING_SLOT,
    RING_SLOTS,
    TWO_HAND,
    ONE_HAND,
    OFF_HAND,
    TOP_K_PER_SLOT,
    _compute_object_scores,
    _normalise_rarete,
    _add_effective_mastery,
    _resolve_weights,
    load_equipements,
)


def _prefer_legendary_variants(group: pd.DataFrame) -> pd.DataFrame:
    """Prefer a legendary row per name when it exists, otherwise keep best available."""
    if group.empty:
        return group
    legend_mask = group["rarete"].map(_normalise_rarete).isin(LEGENDARY_VALUES)
    if not legend_mask.any():
        return group

    chosen_rows = []
    for _, sub in group.groupby("nom"):
        sub = sub.copy()
        if "Score" not in sub.columns:
            sub["Score"] = _compute_object_scores(sub, POIDS_STATS)
        sub = sub.sort_values("Score", ascending=False)
        legends = sub[sub["rarete"].map(_normalise_rarete).isin(LEGENDARY_VALUES)]
        if not legends.empty:
            chosen_rows.append(legends.iloc[0])
        else:
            chosen_rows.append(sub.iloc[0])
    return pd.DataFrame(chosen_rows)


def _build_pools_cp(
    df: pd.DataFrame,
    top_k: int,
    poids: dict[str, float],
    force_legendary: bool,
) -> dict[str, pd.DataFrame]:
    """Build ranked pools similar to GA version, for CP use."""
    if df.empty:
        return {}

    data = df.copy()
    data["Score"] = _compute_object_scores(data, poids)
    pools: dict[str, pd.DataFrame] = {}

    for equipment_type, group in data.groupby("type"):
        equipment_key = str(equipment_type)
        group = group.copy()
        if force_legendary:
            group = _prefer_legendary_variants(group)
        top_group = group.sort_values("Score", ascending=False).head(top_k).copy()
        if equipment_key == RING_SLOT:
            pools[RING_SLOTS[0]] = top_group
            pools[RING_SLOTS[1]] = top_group
        elif equipment_key in {TWO_HAND, ONE_HAND, OFF_HAND} or equipment_key in REQUIRED_SLOTS:
            pools[equipment_key] = top_group

    return pools


def run_optimizer_cp(
    db: Session,
    level: int,
    top_k: int = TOP_K_PER_SLOT,
    force_legendary: bool = False,
    target_stats: Iterable[str] | None = None,
    effective_mastery: Iterable[str] | None = None,
    effective_weight: float = 10.0,
    zero_component_weights: bool = False,
    require_epic: bool = False,
    require_relic: bool = False,
    verbose: bool = False,
) -> tuple[list[dict], dict[str, float], float]:
    """Solve the equipment optimisation with CP-SAT (deterministic)."""
    poids = _resolve_weights(target_stats, POIDS_STATS)

    df_all = load_equipements(db, level_max=level)
    if df_all.empty:
        return [], {}, 0.0

    df_all, poids = _add_effective_mastery(
        df_all,
        effective_mastery,
        weight=effective_weight,
        zero_component_weights=zero_component_weights,
        poids=poids,
    )

    pools = _build_pools_cp(df_all, top_k, poids, force_legendary=force_legendary)
    # Vérifie que tous les slots requis sont présents
    for slot in REQUIRED_SLOTS:
        if slot not in pools or pools[slot].empty:
            return [], {}, 0.0
    if RING_SLOTS[0] not in pools or pools[RING_SLOTS[0]].empty:
        return [], {}, 0.0

    model = cp_model.CpModel()
    selected: dict[str, list[cp_model.IntVar]] = {}

    def add_slot(slot: str, pool: pd.DataFrame, required: bool = True) -> None:
        vars_slot = []
        for idx in pool.index:
            var = model.NewBoolVar(f"{slot}_{idx}")
            vars_slot.append(var)
        if required:
            model.Add(sum(vars_slot) == 1)
        else:
            model.Add(sum(vars_slot) <= 1)
        selected[slot] = vars_slot

    for slot in REQUIRED_SLOTS:
        add_slot(slot, pools[slot])

    ring_pool = pools[RING_SLOTS[0]]
    add_slot(RING_SLOTS[0], ring_pool)
    add_slot(RING_SLOTS[1], ring_pool)

    if TWO_HAND in pools:
        add_slot(TWO_HAND, pools[TWO_HAND], required=False)
    if ONE_HAND in pools:
        add_slot(ONE_HAND, pools[ONE_HAND], required=False)
    if OFF_HAND in pools:
        add_slot(OFF_HAND, pools[OFF_HAND], required=False)

    # Anneaux différents si au moins 2 candidats
    if len(ring_pool.index) >= 2:
        # sum of id * bools for each ring, enforce difference
        ids = [int(i) for i in ring_pool.index.to_list()]
        ring1_val = model.NewIntVar(min(ids), max(ids), "ring1_val")
        ring2_val = model.NewIntVar(min(ids), max(ids), "ring2_val")
        model.Add(ring1_val == sum(idv * var for idv, var in zip(ids, selected[RING_SLOTS[0]])))
        model.Add(ring2_val == sum(idv * var for idv, var in zip(ids, selected[RING_SLOTS[1]])))
        model.Add(ring1_val != ring2_val)

    # Armes : soit 2M, soit 1M+offhand (si disponibles)
    if TWO_HAND in selected or ONE_HAND in selected or OFF_HAND in selected:
        has_two = model.NewBoolVar("has_two_hand")
        has_one = model.NewBoolVar("has_one_hand")
        has_off = model.NewBoolVar("has_off_hand")
        if TWO_HAND in selected:
            model.Add(has_two == sum(selected[TWO_HAND]))
        else:
            model.Add(has_two == 0)
        if ONE_HAND in selected:
            model.Add(has_one == sum(selected[ONE_HAND]))
        else:
            model.Add(has_one == 0)
        if OFF_HAND in selected:
            model.Add(has_off == sum(selected[OFF_HAND]))
        else:
            model.Add(has_off == 0)

        # au moins une arme si un pool existe
        model.Add(has_two + has_one + has_off >= 1)
        # pas de mélange 2M avec autre
        model.Add(has_two + has_one <= 1)
        model.Add(has_two + has_off <= 1)
        # cohérence 1M et offhand : égaux si les deux pools existent
        if ONE_HAND in selected and OFF_HAND in selected:
            model.Add(has_one == has_off)

    # Rareté : max 1 épique et max 1 relique
    def count_rarity(var_map: dict[str, list[cp_model.IntVar]], rarity: str) -> cp_model.LinearExpr | int:
        total_terms = []
        for slot, vars_slot in var_map.items():
            pool = pools[slot if slot not in (RING_SLOTS[1],) else RING_SLOTS[0]]
            for var, idx in zip(vars_slot, pool.index):
                row = pool.loc[idx]
                if str(row.get("rarete", "")) == rarity:
                    total_terms.append(var)
        return sum(total_terms) if total_terms else 0

    epic_count = count_rarity(selected, "Epique")
    relic_count = count_rarity(selected, "Relique")
    model.Add(epic_count <= 1)
    model.Add(relic_count <= 1)
    if require_epic:
        model.Add(epic_count >= 1)
    if require_relic:
        model.Add(relic_count >= 1)

    # Objectif : somme des poids * stats
    objective_terms = []
    all_slots = list(selected.keys())
    for slot in all_slots:
        pool = pools[slot if slot not in (RING_SLOTS[1],) else RING_SLOTS[0]]
        for var, idx in zip(selected[slot], pool.index):
            row = pool.loc[idx]
            row_stats = row.drop(labels=list(META_COLS), errors="ignore")
            score = 0.0
            for stat, weight in poids.items():
                try:
                    val = float(row_stats.get(stat, 0.0) or 0.0)
                except Exception:
                    val = 0.0
                score += max(val, 0.0) * weight
            objective_terms.append(var * score)

    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 8
    if verbose:
        solver.parameters.log_search_progress = True
        solver.parameters.log_to_stdout = True
        solver.parameters.cp_model_probing_level = 1
        solver.parameters.cp_model_presolve = True

    result = solver.Solve(model)
    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], {}, 0.0
    if verbose:
        print("[CP] status:", solver.StatusName(result))
        print("[CP] objective:", solver.ObjectiveValue())
        print("[CP] stats:\n", solver.ResponseStats())

    # Reconstruire le build
    build: list[dict] = []
    stats_sum: dict[str, float] = {}
    for slot in all_slots:
        pool = pools[slot if slot not in (RING_SLOTS[1],) else RING_SLOTS[0]]
        for var, idx in zip(selected[slot], pool.index):
            if solver.Value(var):
                row = pool.loc[idx]
                build.append(dict(row))
                for col, val in row.drop(labels=list(META_COLS), errors="ignore").items():
                    try:
                        stats_sum[col] = stats_sum.get(col, 0.0) + float(val or 0.0)
                    except Exception:
                        pass

    # Score final
    total_score = solver.ObjectiveValue()
    return build, stats_sum, total_score
