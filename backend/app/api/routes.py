import json
import time
import os
import shutil
import joblib
import numpy as np
import uuid
import asyncio

from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import get_db, SessionLocal
from app.db.models import Upload
from app.ml.inference import predict_from_csv, stream_predictions

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load models
lr_model = joblib.load("models/logistic_regression_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")

LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

class PredictRequest(BaseModel):
    features: list[float]
    model: str

@router.get("/")
def home():
    return {"message": "Backend Running 🚀"}

@router.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.features).reshape(1, -1)

    model = rf_model if req.model == "Random Forest" else lr_model

    proba = model.predict_proba(X)[0]
    pred = LABELS[np.argmax(proba)]

    return {
        "activity": pred,
        "confidence": float(np.max(proba)),
        "probabilities": dict(zip(LABELS, proba.tolist()))
    }

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Store in DB
    new_upload = Upload(
        filename=file.filename,
        filepath=file_path,
        prediction=None
    )

    db.add(new_upload)
    db.commit()
    db.refresh(new_upload)

    # 🔥 Run ML model
    try:
        pred_class, confidence = predict_from_csv(file_path)
        label = LABELS[pred_class]

        new_upload.prediction = f"{label} ({confidence:.2f})"
        db.commit()

    except Exception as e:
        new_upload.prediction = f"Error: {str(e)}"
        db.commit()

    return {
        "message": "File uploaded and processed",
        "upload_id": new_upload.id,
        "prediction": new_upload.prediction
    }

@router.post("/upload-stream")
async def upload_stream(
    file: UploadFile = File(...), 
    window_size: int = 5, 
    delay: float = 0.3, 
    db: Session = Depends(get_db)
):
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Store in DB
    new_upload = Upload(
        filename=file.filename,
        filepath=file_path,
        prediction="Streaming..."
    )

    db.add(new_upload)
    db.commit()
    db.refresh(new_upload)

    # ✅ Changed to async generator
    async def event_stream():
        try:
            yield json.dumps({"status": "started", "upload_id": new_upload.id}) + "\n"

            last_prediction = None

            for pred_class, confidence in stream_predictions(file_path, window_size):
                print("STREAM OUTPUT:", pred_class, confidence)  # 🔥 ADD THIS
                label = LABELS[pred_class]

                last_prediction = f"{label} ({confidence:.2f})"

                data = {
                    "activity": label,
                    "confidence": confidence,
                    "timestamp": time.time()
                }

                yield json.dumps(data) + "\n"
                
                # ✅ Async-safe sleep
                await asyncio.sleep(delay)

            # After streaming ends → update DB with a fresh session
            if last_prediction:
                session = SessionLocal()
                try:
                    obj = session.query(Upload).filter(Upload.id == new_upload.id).first()
                    if obj:
                        obj.prediction = last_prediction
                        session.commit()
                finally:
                    session.close()
            
            # ✅ Stream termination signal
            yield json.dumps({"status": "completed"}) + "\n"

        except Exception as e:
            # Handle errors safely with a fresh session
            error_session = SessionLocal()
            try:
                obj = error_session.query(Upload).filter(Upload.id == new_upload.id).first()
                if obj:
                    obj.prediction = f"Error: {str(e)}"
                    error_session.commit()
            finally:
                error_session.close()
                
            yield json.dumps({"error": str(e)}) + "\n"

    # ✅ Updated media_type to standard SSE
    return StreamingResponse(event_stream(), media_type="text/event-stream")