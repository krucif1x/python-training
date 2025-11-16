# src/api/db.py
import os
from datetime import datetime

from sqlalchemy import (
    create_engine, MetaData, Table,
    Column, Integer, String, Float, DateTime
)
from sqlalchemy.orm import sessionmaker

# DATABASE URL:
# For PostgreSQL use:
# export DATABASE_URL="postgresql://user:pass@localhost:5432/anpr"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./anpr.db")

# SQLAlchemy engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Metadata
meta = MetaData()

# Table definition
plate_logs = Table(
    "plate_logs", meta,
    Column("id", Integer, primary_key=True, index=True),
    Column("plate_text", String(64)),
    Column("confidence", Float),
    Column("detector_conf", Float),
    Column("ocr_conf", Float),
    Column("nlp_score", Float),
    Column("camera_id", String(64)),
    Column("timestamp", DateTime, default=datetime.utcnow),
    Column("image_path", String(255)),
)

# Create tables if not exist
meta.create_all(engine)

# Session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ================================
# CRUD FUNCTIONS
# ================================

def insert_plate_log(
    plate_text: str,
    confidence: float,
    detector_conf: float,
    ocr_conf: float,
    nlp_score: float,
    camera_id: str,
    image_path: str = None
):
    """Insert a plate recognition log into DB."""
    db = SessionLocal()
    try:
        insert_query = plate_logs.insert().values(
            plate_text=plate_text,
            confidence=confidence,
            detector_conf=detector_conf,
            ocr_conf=ocr_conf,
            nlp_score=nlp_score,
            camera_id=camera_id,
            timestamp=datetime.utcnow(),
            image_path=image_path,
        )
        db.execute(insert_query)
        db.commit()
    finally:
        db.close()


def get_latest_logs(limit: int = 50):
    """Get latest N logs for dashboard."""
    db = SessionLocal()
    try:
        query = plate_logs.select().order_by(plate_logs.c.timestamp.desc()).limit(limit)
        rows = db.execute(query).fetchall()
        return [dict(r._mapping) for r in rows]
    finally:
        db.close()
