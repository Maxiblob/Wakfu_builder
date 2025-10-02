#!/usr/bin/env python

import pandas as pd
from sqlalchemy import create_engine
import os

DATABASE_URL = os.environ["DATABASE_URL"]
TABLE_NAME = "wakfu_equipement"

# -------------------------

csv_file = "app/db/items_wakfu_complet_final_utf8.csv"
df = pd.read_csv(csv_file)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(0)
    
# -------------------------

engine = create_engine(DATABASE_URL)

# -------------------------

insert = df.to_sql(TABLE_NAME, engine, if_exists="append", index=True, index_label="id")

print(f"Succès : données insérées dans la table '{TABLE_NAME} - {insert} lignes ajoutées.'")