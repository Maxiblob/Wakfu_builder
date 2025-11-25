#!/usr/bin/env python

import os
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv

# Charge les variables d'environnement depuis .env (recherche ascendante)
load_dotenv(find_dotenv())

DATABASE_URL = os.environ["DATABASE_URL"]
TABLE_NAME = "wakfu_equipement"
if_exists_env = os.getenv("DB_IF_EXISTS", "replace").lower()
_allowed_if_exists = ("fail", "replace", "append")
if if_exists_env not in _allowed_if_exists:
    if_exists_env = "replace"
IF_EXISTS: Literal["fail", "replace", "append"] = cast(Literal["fail", "replace", "append"], if_exists_env)

# -------------------------

csv_path = Path(__file__).resolve().parent / "items_wakfu_complet_final_utf8.csv"
if not csv_path.is_file():
    raise FileNotFoundError(f"CSV introuvable: {csv_path}")

df = pd.read_csv(csv_path)
# Conserver une colonne id stable basée sur l'ordre du CSV
df = df.reset_index().rename(columns={"index": "id"})
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(0)
    
# -------------------------

engine = create_engine(DATABASE_URL)

# -------------------------

inserted = df.to_sql(
    TABLE_NAME,
    engine,
    if_exists=IF_EXISTS,  # replace by default to éviter les doublons/échecs partiels
    index=False,          # on fournit déjà la colonne id
)

print(f"Succès : {inserted} lignes écrites dans la table '{TABLE_NAME}' (if_exists={IF_EXISTS}).")
