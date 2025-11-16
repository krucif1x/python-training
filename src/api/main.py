# src/api/main.py
from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData, DateTime
from datetime import datetime

app = FastAPI()
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./anpr.db")
engine = create_engine(DB_URL, echo=False)
meta = MetaData()
plate_logs = Table(
    'plate_logs', meta,
    Column('id', Integer, primary_key=True),
    Column('plate_text', String),
    Column('confidence', Float),
    Column('detector_conf', Float),
    Column('ocr_conf', Float),
    Column('nlp_score', Float),
    Column('camera_id', String),
    Column('timestamp', DateTime),
    Column('image_path', String),
)
meta.create_all(engine)

@app.post("/log_plate/")
async def log_plate(plate_text: str = Form(...), confidence: float = Form(...),
                    detector_conf: float = Form(...), ocr_conf: float = Form(...),
                    nlp_score: float = Form(...), camera_id: str = Form("cam0"),
                    image: UploadFile = File(None)):
    img_path = None
    if image:
        saved = f"logs/{datetime.utcnow().isoformat()}_{image.filename}"
        os.makedirs(os.path.dirname(saved), exist_ok=True)
        with open(saved,"wb") as f:
            shutil.copyfileobj(image.file, f)
        img_path = saved
    ins = plate_logs.insert().values(
        plate_text=plate_text,
        confidence=confidence,
        detector_conf=detector_conf,
        ocr_conf=ocr_conf,
        nlp_score=nlp_score,
        camera_id=camera_id,
        timestamp=datetime.utcnow(),
        image_path=img_path
    )
    conn = engine.connect()
    conn.execute(ins)
    conn.close()
    return {"status":"ok", "plate": plate_text}
