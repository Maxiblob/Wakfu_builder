from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable

import pandas as pd
from sqlalchemy.orm import Session

from app.models import wakfu_equipement

TEXT_COLS = {"nom", "type", "rarete", "effets_supplementaires"}
META_COLS = TEXT_COLS | {"niveau", "id"}

REQUIRED_SLOTS = ["Casque", "Amulette", "Plastron", "Bottes", "Cape", "Epaulettes", "Ceinture"]
BONUS_SLOTS = ["Emblème", "Familier"]
RING_SLOT = "Anneau"
RING_SLOTS = ("Anneau1", "Anneau2")

TWO_HAND = "Arme 2 Mains"
ONE_HAND = "Arme 1 Main"
OFF_HAND = "Seconde Main"

TOP_K_PER_SLOT = 25
LEGENDARY_VALUES = {"légendaire", "legendaire", "lÃ©gendaire"}

POIDS_STATS: dict[str, float] = {
    "pa": 600,
    "pm": 600,
    "pw": 100,
    "portee": 600,
    "controle": 200,
    "pv": 1,
    "coup_critique": 5,
    "maitrise_melee": 0.75,
    "maitrise_distance": 0.75,
    "maitrise_berserk": 0.75,
    "maitrise_critique": 0.75,
    "maitrise_dos": 0.75,
    "maitrise_1_element": 0.75,
    "maitrise_2_elements": 1,
    "maitrise_3_elements": 1.25,
    "maitrise_elementaire": 10,
    "maitrise_feu": 0.75,
    "maitrise_eau": 0.75,
    "maitrise_terre": 0.75,
    "maitrise_air": 0.75,
    "maitrise_soin": 0.75,
    "tacle": 0.5,
    "esquive": 0.5,
    "initiative": 5,
    "parade": 5,
    "resistance_elementaire": 10,
    "resistance_1_element": 2.5,
    "resistance_2_elements": 5,
    "resistance_3_elements": 7.5,
    "resistance_feu": 2.5,
    "resistance_eau": 2.5,
    "resistance_terre": 2.5,
    "resistance_air": 2.5,
    "resistance_critique": 7.5,
    "resistance_dos": 7.5,
    "armure_donnee": 5,
    "armure_recue": 5,
    "volonte": 5,
}

StatWeights = Mapping[str, float]
SlotIndices = dict[str, int]
BuildStats = dict[str, float]
Pools = dict[str, pd.DataFrame]


def _resolve_weights(target_stats: Iterable[str] | None, base_weights: StatWeights) -> dict[str, float]:
    """Return a weight map prioritising only the requested stats when provided."""
    if not target_stats:
        return dict(base_weights)

    normalized = {str(stat).strip().lower() for stat in target_stats if str(stat).strip()}
    resolved: dict[str, float] = {}
    for stat, weight in base_weights.items():
        resolved[stat] = float(weight) if stat in normalized else 0.0

    # If a stat name isn't known, still allow it with a neutral weight
    for extra in normalized - set(base_weights):
        resolved[extra] = 1.0
    return resolved


def _add_effective_mastery(
    df: pd.DataFrame,
    fields: Iterable[str] | None,
    weight: float,
    zero_component_weights: bool,
    poids: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add a derived 'maitrise_effective' column from selected mastery fields."""
    if not fields:
        return df, poids

    candidates = [f for f in fields if f in df.columns]
    if not candidates:
        return df, poids

    df = df.copy()
    df["maitrise_effective"] = (
        df[candidates]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
    )

    new_weights = dict(poids)
    if zero_component_weights:
        for stat in candidates:
            if stat in new_weights:
                new_weights[stat] = 0.0
    new_weights["maitrise_effective"] = float(weight)
    return df, new_weights


def _safe_float(value: Any) -> float:
    """Coerce any value to a finite float, falling back to zero."""
    if isinstance(value, Real):
        result = float(value)
        return 0.0 if math.isnan(result) else result
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(result) else result


def _choose_random_index(pool: pd.DataFrame) -> int:
    """Return a random index from a non-empty equipment pool."""
    indices = [int(idx) for idx in pool.index.to_list()]
    if not indices:
        raise ValueError("Cannot sample from an empty pool")
    return int(random.choice(indices))


def _normalise_ring_selection(selection: SlotIndices, ring_pool: pd.DataFrame | None) -> None:
    """Ensure the ring selection contains two valid indices when possible."""
    if ring_pool is None or ring_pool.empty:
        selection.pop(RING_SLOTS[0], None)
        selection.pop(RING_SLOTS[1], None)
        return

    ring_indices = [int(idx) for idx in ring_pool.index.to_list()]
    if not ring_indices:
        selection.pop(RING_SLOTS[0], None)
        selection.pop(RING_SLOTS[1], None)
        return

    if len(ring_indices) == 1:
        single_index = ring_indices[0]
        selection[RING_SLOTS[0]] = single_index
        selection[RING_SLOTS[1]] = single_index
        return

    first = selection.get(RING_SLOTS[0])
    if first not in ring_indices:
        first = random.choice(ring_indices)
        selection[RING_SLOTS[0]] = int(first)

    second = selection.get(RING_SLOTS[1])
    if second not in ring_indices or second == selection[RING_SLOTS[0]]:
        alternatives = [idx for idx in ring_indices if idx != selection[RING_SLOTS[0]]]
        if alternatives:
            second = random.choice(alternatives)
        else:
            second = selection[RING_SLOTS[0]]
        selection[RING_SLOTS[1]] = int(second)


def _normalise_weapon_selection(selection: SlotIndices, pools: Pools) -> None:
    """Keep only compatible weapon combinations in the selection."""
    has_two_hand = TWO_HAND in selection
    has_one_hand = ONE_HAND in selection
    has_off_hand = OFF_HAND in selection

    if has_two_hand and (has_one_hand or has_off_hand):
        selection.pop(ONE_HAND, None)
        selection.pop(OFF_HAND, None)
        return

    if has_one_hand != has_off_hand:
        if has_one_hand and OFF_HAND in pools and not pools[OFF_HAND].empty:
            selection[OFF_HAND] = _choose_random_index(pools[OFF_HAND])
        elif has_off_hand and ONE_HAND in pools and not pools[ONE_HAND].empty:
            selection[ONE_HAND] = _choose_random_index(pools[ONE_HAND])
        else:
            selection.pop(ONE_HAND, None)
            selection.pop(OFF_HAND, None)


def _normalise_selection(selection: SlotIndices, pools: Pools) -> None:
    """Normalise ring and weapon choices in place."""
    _normalise_ring_selection(selection, pools.get(RING_SLOTS[0]))
    _normalise_weapon_selection(selection, pools)


def _get_item_row(df: pd.DataFrame, idx: int) -> pd.Series:
    """Fetch a row by index, returning a Series even when duplicates exist."""
    row = df.loc[idx]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _compute_object_scores(df: pd.DataFrame, poids: StatWeights) -> pd.Series:
    """Vectorised score computation for a whole equipment dataframe."""
    if df.empty:
        return pd.Series(dtype=float)
    stats = [stat for stat in poids if stat in df.columns]
    if not stats:
        return pd.Series(0.0, index=df.index, dtype=float)
    stats_df = df[stats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    weights = pd.Series(poids, dtype=float).reindex(stats).fillna(0.0)
    positive_stats = stats_df.clip(lower=0.0)
    return positive_stats.mul(weights, axis=1).sum(axis=1)


def load_equipements(
    db: Session,
    level_max: int | None = None,
    ban_ids: Iterable[int] | None = None,
    ban_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load equipment entries from the database as a dataframe, filtered and deduped."""
    query = db.query(wakfu_equipement)
    if level_max is not None and level_max > 0:
        query = query.filter(wakfu_equipement.niveau <= level_max)

    items = query.all()
    if not items:
        return pd.DataFrame()

    records = [
        {col.name: getattr(item, col.name) for col in wakfu_equipement.__table__.columns}
        for item in items
    ]
    df = pd.DataFrame.from_records(records)

    # Dedupe sur id si présent, sinon sur (nom, type, niveau)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    else:
        df = df.drop_duplicates(subset=["nom", "type", "niveau"], keep="first")

    if ban_ids and "id" in df.columns:
        df = df[~df["id"].isin(list(ban_ids))]
    if ban_names and "nom" in df.columns:
        ban_norm = {str(n).strip().lower() for n in ban_names}
        df = df[df["nom"].str.strip().str.lower().map(lambda x: x not in ban_norm)]

    df = df.reset_index(drop=True)
    return df


def score_objet(row: pd.Series, poids: StatWeights, allow_neg_score: bool = False) -> float:
    """Compute a weighted score for one equipment piece."""
    score = 0.0
    for stat, weight in poids.items():
        value = _safe_float(row.get(stat, 0.0))
        if not allow_neg_score:
            value = max(value, 0.0)
        score += value * weight
    return score


def score_objet_brut(row: pd.Series, poids: StatWeights = POIDS_STATS) -> float:
    """Compute a raw sum of positive stats for one equipment piece."""
    return sum(max(_safe_float(row.get(stat, 0.0)), 0.0) for stat in poids)


def score_build(df: pd.DataFrame, indices: SlotIndices, poids: StatWeights) -> tuple[float, BuildStats]:
    """Aggregate equipment stats and return (weighted_score, raw_stats)."""
    if not indices:
        return 0.0, {}
    selected_rows = df.loc[list(indices.values())].copy()
    if isinstance(selected_rows, pd.Series):
        selected_rows = selected_rows.to_frame().T
    stats_df = selected_rows.drop(columns=list(META_COLS), errors="ignore")
    stats_sum: BuildStats = {}
    if not stats_df.empty:
        numeric_stats = stats_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        totals = numeric_stats.sum(axis=0)
        stats_sum = {str(column): float(value) for column, value in totals.items()}
    score = sum(max(stats_sum.get(stat, 0.0), 0.0) * weight for stat, weight in poids.items())
    return score, stats_sum


def est_build_valide(
    df: pd.DataFrame,
    indices: SlotIndices,
    require_epic: bool = False,
    require_relic: bool = False,
) -> tuple[bool, str]:
    """Validate slot coverage, ring uniqueness, rarity limits, weapon pairing, and optional relic/epic presence."""
    if not indices:
        return False, "No equipment selected"

    try:
        items = {slot: _get_item_row(df, idx) for slot, idx in indices.items()}
    except KeyError as exc:
        return False, f"Missing equipment index: {exc.args[0]}"

    for slot in REQUIRED_SLOTS:
        if slot not in items:
            return False, f"Missing required slot: {slot}"

    ring_items = [items.get(slot) for slot in RING_SLOTS if slot in items]
    # Ensure both ring slots are present and non-None
    if len(ring_items) != len(RING_SLOTS) or any(item is None for item in ring_items):
        return False, "Exactly two rings are required"
    ring_a, ring_b = ring_items
    # Safely extract an identifier, preferring "id" then "nom"
    id_a = ring_a.get("id", ring_a.get("nom")) if ring_a is not None else None
    id_b = ring_b.get("id", ring_b.get("nom")) if ring_b is not None else None
    if id_a == id_b:
        return False, "Rings must be different"

    rarities = [str(row.get("rarete", "")) for row in items.values()]
    if sum(r == "Epique" for r in rarities) > 1:
        return False, "At most one epic item is allowed"
    if sum(r == "Relique" for r in rarities) > 1:
        return False, "At most one relic item is allowed"

    if require_epic and not any(r == "Epique" for r in rarities):
        return False, "At least one epic item is required"
    if require_relic and not any(r == "Relique" for r in rarities):
        return False, "At least one relic item is required"

    weapon_slots = [slot for slot in items if slot.startswith("Arme") or slot.startswith("Seconde")]
    weapon_types = [str(items[slot].get("type", "")) for slot in weapon_slots]
    has_two_hand = any(TWO_HAND in weapon_type for weapon_type in weapon_types)
    has_one_hand = any(ONE_HAND in weapon_type for weapon_type in weapon_types)
    has_off_hand = any(OFF_HAND in weapon_type for weapon_type in weapon_types)

    if has_two_hand and (has_one_hand or has_off_hand):
        return False, "Cannot mix two handed and one handed weapons"
    if not has_two_hand and not (has_one_hand and has_off_hand):
        return False, "Select either one two handed weapon or a main hand plus off hand"

    return True, "OK"

def _normalise_rarete(value: Any) -> str:
    return str(value).strip().lower()


def _prefer_legendary_variants(group: pd.DataFrame) -> pd.DataFrame:
    """When force_legendary is enabled, prefer legendary rows per item name if available."""
    if group.empty:
        return group
    legend_mask = group["rarete"].map(_normalise_rarete).isin(LEGENDARY_VALUES)
    if not legend_mask.any():
        return group

    # Pour chaque nom, si une version légendaire existe on la garde, sinon on garde la meilleure
    chosen_rows = []
    for name, sub in group.groupby("nom"):
        sub_with_score = sub.copy()
        if "Score" not in sub_with_score.columns:
            sub_with_score["Score"] = _compute_object_scores(sub_with_score, POIDS_STATS)
        sub_with_score = sub_with_score.sort_values("Score", ascending=False)
        legend_subset = sub_with_score[sub_with_score["rarete"].map(_normalise_rarete).isin(LEGENDARY_VALUES)]
        if not legend_subset.empty:
            chosen_rows.append(legend_subset.iloc[0])
        else:
            chosen_rows.append(sub_with_score.iloc[0])
    return pd.DataFrame(chosen_rows)


def build_pools(
    df: pd.DataFrame,
    top_k: int = TOP_K_PER_SLOT,
    poids: StatWeights = POIDS_STATS,
    force_legendary: bool = False,
) -> Pools:
    """Build ranked equipment pools per slot to reduce combinatorics."""
    if df.empty:
        return {}

    data = df.copy()
    data["Score"] = _compute_object_scores(data, poids)
    pools: Pools = {}

    for equipment_type, group in data.groupby("type"):
        # Ensure the group key is a string (pandas group keys can be non-str like dates/timestamps)
        equipment_key = str(equipment_type)
        group = group.copy()
        if force_legendary:
            # Préfère les versions légendaires pour un même nom si disponibles,
            # sans exclure les autres raretés (épique/relique) si aucune version légendaire n'existe.
            group = _prefer_legendary_variants(group)

        top_group = group.sort_values("Score", ascending=False).head(top_k).copy()
        if equipment_key == RING_SLOT:
            pools[RING_SLOTS[0]] = top_group
            pools[RING_SLOTS[1]] = top_group
        elif equipment_key in {TWO_HAND, ONE_HAND, OFF_HAND} or equipment_key in REQUIRED_SLOTS:
            pools[equipment_key] = top_group

    missing = [slot for slot in REQUIRED_SLOTS if slot not in pools]
    if missing:
        print(f"[WARNING] Missing pools for slots: {missing}")

    # Nettoyage : retire les pools vides et borne les scores négatifs à zéro
    for key, pool in list(pools.items()):
        if pool.empty:
            pools.pop(key, None)
            continue
        numeric_cols = pool.select_dtypes(include=["number"]).columns
        pool[numeric_cols] = pool[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        pool[numeric_cols] = pool[numeric_cols].clip(lower=0.0)
        pools[key] = pool

    return pools


def random_build(pools: Pools) -> SlotIndices:
    """Generate a random build by sampling one item per slot."""
    selection: SlotIndices = {}

    for slot in REQUIRED_SLOTS:
        pool = pools.get(slot)
        if pool is not None and not pool.empty:
            selection[slot] = _choose_random_index(pool)

    ring_pool = pools.get(RING_SLOTS[0])
    if ring_pool is not None and not ring_pool.empty:
        ring_indices = [int(idx) for idx in ring_pool.index.to_list()]
        if ring_indices:
            if len(ring_indices) >= 2:
                chosen = random.sample(ring_indices, k=2)
            else:
                chosen = ring_indices * 2
            selection[RING_SLOTS[0]], selection[RING_SLOTS[1]] = chosen[0], chosen[1]

    weapon_two = pools.get(TWO_HAND)
    weapon_one = pools.get(ONE_HAND)
    off_hand = pools.get(OFF_HAND)
    if weapon_two is not None and not weapon_two.empty and random.random() < 0.5:
        selection[TWO_HAND] = _choose_random_index(weapon_two)
    elif (
        weapon_one is not None
        and not weapon_one.empty
        and off_hand is not None
        and not off_hand.empty
    ):
        selection[ONE_HAND] = _choose_random_index(weapon_one)
        selection[OFF_HAND] = _choose_random_index(off_hand)

    _normalise_selection(selection, pools)
    return selection


@dataclass(slots=True)
class Individual:
    genes: SlotIndices
    fitness: float


def evaluate(
    df: pd.DataFrame,
    indiv: Individual,
    poids: StatWeights,
    require_epic: bool = False,
    require_relic: bool = False,
) -> float:
    """Evaluate an individual build by returning its weighted score."""
    is_valid, _ = est_build_valide(df, indiv.genes, require_epic=require_epic, require_relic=require_relic)
    if not is_valid:
        return float("-inf")
    score, _ = score_build(df, indiv.genes, poids)
    return score


def crossover(parent1: Individual, parent2: Individual, pools: Pools, rate: float = 0.5) -> SlotIndices:
    """Combine genes from two parents to create a child selection."""
    child: SlotIndices = {}
    slots = set(parent1.genes) | set(parent2.genes)
    for slot in slots:
        if random.random() < rate and slot in parent1.genes:
            child[slot] = parent1.genes[slot]
        elif slot in parent2.genes:
            child[slot] = parent2.genes[slot]
    _normalise_selection(child, pools)
    return child


def mutate(child: SlotIndices, pools: Pools, proba: float = 0.2) -> SlotIndices:
    """Mutate a child selection by resampling some slots."""
    mutated = child.copy()
    for slot, pool in pools.items():
        if slot == RING_SLOTS[1] or pool.empty:
            continue
        if random.random() >= proba:
            continue
        if slot == RING_SLOTS[0]:
            ring_indices = [int(idx) for idx in pool.index.to_list()]
            if not ring_indices:
                continue
            if len(ring_indices) >= 2:
                chosen = random.sample(ring_indices, k=2)
            else:
                chosen = ring_indices * 2
            mutated[RING_SLOTS[0]], mutated[RING_SLOTS[1]] = chosen[0], chosen[1]
            continue
        mutated[slot] = _choose_random_index(pool)
    _normalise_selection(mutated, pools)
    return mutated


def run_algo_gen(
    df: pd.DataFrame,
    pools: Pools,
    pop_size: int = 60,
    generations_max: int = 80,
    elite: int = 3,
    proba_mutation: float = 0.2,
    verbose: bool = True,
    stagnation_patience: int = 12,
    poids: StatWeights = POIDS_STATS,
    require_epic: bool = False,
    require_relic: bool = False,
) -> Individual:
    """Run a simple genetic algorithm to search for a high scoring build."""
    if pop_size <= 0:
        raise ValueError("Population size must be positive")
    if pop_size < 2:
        raise ValueError("Population size must be at least 2")
    if elite <= 0 or elite > pop_size:
        raise ValueError("Elite count must be between 1 and population size")
    if df.empty or not pools:
        raise ValueError("Cannot run genetic algorithm without data")

    population: list[Individual] = []
    for _ in range(pop_size):
        genes = random_build(pools)
        individual = Individual(genes=genes, fitness=0.0)
        individual.fitness = evaluate(df, individual, poids, require_epic=require_epic, require_relic=require_relic)
        population.append(individual)

    best_score_global = float("-inf")
    stagnation_steps = 0

    for generation in range(generations_max):
        population.sort(key=lambda individual: individual.fitness, reverse=True)
        new_population: list[Individual] = population[:elite]

        mating_pool_size = min(len(population), max(2, max(10, pop_size // 2)))
        parents_pool = population[:mating_pool_size]

        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(parents_pool, 2)
            child_genes = crossover(parent1, parent2, pools)
            child_genes = mutate(child_genes, pools, proba_mutation)
            child = Individual(genes=child_genes, fitness=0.0)
            child.fitness = evaluate(df, child, poids, require_epic=require_epic, require_relic=require_relic)
            new_population.append(child)

        population = new_population

        # Anti-stagnation: si aucun progrès pendant plusieurs générations, réinjecter de l'aléatoire
        current_best = population[0].fitness
        if current_best > best_score_global:
            best_score_global = current_best
            stagnation_steps = 0
        else:
            stagnation_steps += 1
            if stagnation_steps >= stagnation_patience:
                # Remplacer ~30% de la population par de nouveaux individus
                replace_count = max(1, pop_size // 3)
                for _ in range(replace_count):
                    genes = random_build(pools)
                    indiv = Individual(
                        genes=genes,
                        fitness=evaluate(
                            df,
                            Individual(genes=genes, fitness=0.0),
                            poids,
                            require_epic=require_epic,
                            require_relic=require_relic,
                        ),
                    )
                    population.append(indiv)
                population.sort(key=lambda individual: individual.fitness, reverse=True)
                population = population[:pop_size]
                stagnation_steps = 0

        if verbose:
            best_individual = population[0]
            print(f"Gen {generation + 1}/{generations_max} - best score: {best_individual.fitness:.2f}")

    population.sort(key=lambda individual: individual.fitness, reverse=True)
    return population[0]


def run_optimizer(
    db: Session,
    level: int,
    top_k: int = TOP_K_PER_SLOT,
    pop_size: int = 60,
    generations_max: int = 80,
    elite: int = 3,
    proba_mutation: float = 0.2,
    verbose: bool = True,
    force_legendary: bool = False,
    target_stats: Iterable[str] | None = None,
    effective_mastery: Iterable[str] | None = None,
    effective_weight: float = 10.0,
    zero_component_weights: bool = False,
    require_epic: bool = False,
    require_relic: bool = False,
    prioritize_pa: bool = False,
    prioritize_pm: bool = False,
    ban_ids: Iterable[int] | None = None,
    ban_names: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], BuildStats, float, list[tuple[list[dict[str, Any]], BuildStats, float]]]:
    """Complete optimisation pipeline returning the best build, stats, and score."""
    poids = _resolve_weights(target_stats, POIDS_STATS)

    # Boost PA/PM if explicitly requested
    if prioritize_pa:
        poids["pa"] = max(poids.get("pa", 0.0), 10000.0)
    if prioritize_pm:
        poids["pm"] = max(poids.get("pm", 0.0), 10000.0)

    df_all = load_equipements(db, level_max=level, ban_ids=ban_ids, ban_names=ban_names)
    if df_all.empty:
        return [], {}, 0.0, []

    # Ajout d'une maîtrise effective dérivée (somme des champs fournis)
    df_all, poids = _add_effective_mastery(
        df_all,
        effective_mastery,
        weight=effective_weight,
        zero_component_weights=zero_component_weights,
        poids=poids,
    )

    pools = build_pools(df_all, top_k, poids, force_legendary=force_legendary)
    if not pools:
        return [], {}, 0.0, []

    best = run_algo_gen(
        df_all,
        pools,
        pop_size=pop_size,
        generations_max=generations_max,
        elite=elite,
        proba_mutation=proba_mutation,
        verbose=verbose,
        poids=poids,
        require_epic=require_epic,
        require_relic=require_relic,
    )
    best_score, best_stats = score_build(df_all, best.genes, poids)

    build: list[dict[str, Any]] = []
    for idx in best.genes.values():
        row = _get_item_row(df_all, idx)
        build.append(dict(row))

    return build, best_stats, best_score, []
