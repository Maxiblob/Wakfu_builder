from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import wakfu_equipement

router = APIRouter()

@router.get("/")
def get_items(db: Session = Depends(get_db)):
    items = db.query(wakfu_equipement).limit(50).all()
    return items