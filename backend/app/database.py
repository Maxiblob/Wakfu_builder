from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DB_USER = os.environ["DB_USER"]
DB_PASS = os.environ["DB_PASS"]
DB_NAME = os.environ["DB_NAME"]
DB_HOST = os.environ["DB_HOST"]
DATABASE_URL = os.environ["DATABASE_URL"]

#print(f"Connecting to database at {DB_HOST} with user {DB_USER}, database {DB_NAME}, URL: {DATABASE_URL}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dépendance FastAPI
def get_db():
    db = SessionLocal()
    with db as db:
        yield db
