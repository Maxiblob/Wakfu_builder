from __future__ import annotations

from typing import Any

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

from pydantic import BaseModel

Base = declarative_base()

class Item(BaseModel):
    id: int
    nom: str
    type: str
    rarete: str
    niveau: int
    pa : int | None = 0
    pm : int | None = 0
    pw : int | None = 0
    portee : int | None = 0
    controle : int | None = 0
    pv : int | None = 0
    coup_critique : int | None = 0
    maitrise_melee : int | None = 0
    maitrise_distance : int | None = 0
    maitrise_berserk : int | None = 0
    maitrise_critique : int | None = 0
    maitrise_dos : int | None = 0
    maitrise_1_element : int | None = 0
    maitrise_2_elements : int | None = 0
    maitrise_3_elements : int | None = 0
    maitrise_elementaire : int | None = 0
    maitrise_feu : int | None = 0
    maitrise_eau : int | None = 0
    maitrise_terre : int | None = 0
    maitrise_air : int | None = 0
    maitrise_soin : int | None = 0
    tacle : int | None = 0
    esquive : int | None = 0
    initiative : int | None = 0
    parade : int | None = 0
    resistance_elementaire : int | None = 0
    resistance_1_element : int | None = 0
    resistance_2_elements : int | None = 0
    resistance_3_elements : int | None = 0
    resistance_feu : int | None = 0
    resistance_eau : int | None = 0
    resistance_terre : int | None = 0
    resistance_air : int | None = 0
    resistance_critique : int | None = 0
    resistance_dos : int | None = 0
    armure_donnee : int | None = 0
    armure_recue : int | None = 0
    volonte : int | None = 0
    effets_supplementaires : str | None = ""

class BuildResponse(BaseModel):
    score: float
    items: list[dict[str, Any]]
    stats: dict[str, float]
    alternatives: list["BuildResponse"] | None = None

class wakfu_equipement(Base):
    __tablename__ = "wakfu_equipement"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False, default="")
    type = Column(String, nullable=False, default="")
    rarete = Column(String, nullable=False, default="")
    niveau = Column(Integer, nullable=False, default=0)
    pa = Column(Integer, default=0)
    pm = Column(Integer, default=0)
    pw = Column(Integer, default=0)
    portee = Column(Integer, default=0)
    controle = Column(Integer, default=0)
    pv = Column(Integer, default=0)
    coup_critique = Column(Integer, default=0)
    maitrise_melee = Column(Integer, default=0)
    maitrise_distance = Column(Integer, default=0)
    maitrise_berserk = Column(Integer, default=0)
    maitrise_critique = Column(Integer, default=0)
    maitrise_dos = Column(Integer, default=0)
    maitrise_1_element = Column(Integer, default=0)
    maitrise_2_elements = Column(Integer, default=0)
    maitrise_3_elements = Column(Integer, default=0)
    maitrise_elementaire = Column(Integer, default=0)
    maitrise_feu = Column(Integer, default=0)
    maitrise_eau = Column(Integer, default=0)
    maitrise_terre = Column(Integer, default=0)
    maitrise_air = Column(Integer, default=0)
    maitrise_soin = Column(Integer, default=0)
    tacle = Column(Integer, default=0)
    esquive = Column(Integer, default=0)
    initiative = Column(Integer, default=0)
    parade = Column(Integer, default=0)
    resistance_elementaire = Column(Integer, default=0)
    resistance_1_element = Column(Integer, default=0)
    resistance_2_elements = Column(Integer, default=0)
    resistance_3_elements = Column(Integer, default=0)
    resistance_feu = Column(Integer, default=0)
    resistance_eau = Column(Integer, default=0)
    resistance_terre = Column(Integer, default=0)
    resistance_air = Column(Integer, default=0)
    resistance_critique = Column(Integer, default=0)
    resistance_dos = Column(Integer, default=0)
    armure_donnee = Column(Integer, default=0)
    armure_recue = Column(Integer, default=0)
    volonte = Column(Integer, default=0)
    effets_supplementaires = Column(String)
