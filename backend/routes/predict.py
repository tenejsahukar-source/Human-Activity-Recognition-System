from fastapi import APIRouter, UploadFile, File, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import joblib
import os
import json
import time
import asyncio
import tempfile
from typing import AsyncGenerator

from app.db.database import get_db
from app.db.models import Upload

router = APIRouter()

# ✅ Load model, scaler, and expected columns
model = joblib.load("models/random_forest_model.pkl")

try:
    scaler = joblib.load("models/scaler.pkl")
except:
    scaler = None

try:
    expected_columns = joblib.load("models/columns.pkl")
except:
    expected_columns = None

# ✅ Label mapping
label_map = {
    0: "Walking",
    1: "Sitting",
    2: "Standing",
    3: "Lying"
}

# ✅ Correct upload path (ABSOLUTE)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# 🔥 Upgraded safe_predict function
def safe_predict(model, X, scaler=None):
    if scaler:
        X = scaler.transform(X)

    X = np.nan_to_num(X)
    X = np.clip(X, -1e6, 1e6)

    pred = model.predict(X)

    try:
        probs = model.predict_proba(X)
        confidence = np.max(probs, axis=1)
    except:
        confidence = np.ones(len(pred))

    return pred, confidence


# ==========================================
# 🚀 ENDPOINT 1: BATCH PROCESSING (Memory Safe)
# ==========================================
@router.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # 🔥 FINAL FIX: Chunked writing prevents RAM spikes for huge files
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
                f.write(chunk)

        df = pd.read_csv(file_path)

        if expected_columns:
            df = df.reindex(columns=expected_columns, fill_value=0)

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        data = df.values
        results = []
        start_time = time.time()
        
        for i in range(len(data)):
            row = data[i].reshape(1, -1)
            
            # 🔥 Shape Drift Protection (Soft Fail)
            if scaler and hasattr(scaler, 'n_features_in_'):
                if row.shape[1] != scaler.n_features_in_:
                    print(f"⚠️ Batch Feature mismatch: Expected {scaler.n_features_in_}, got {row.shape[1]}")
                    continue

            pred, conf = safe_predict(model, row, scaler)
            results.append({
                "activity": label_map.get(int(pred[0]), "Unknown"),
                "confidence": float(conf[0])
            })
            
        end_time = time.time()

        mapped = results

        # Save to DB
        record = Upload(
            filename=file.filename,
            filepath=file_path,
            prediction=json.dumps(mapped) 
        )
        db.add(record)
        
        # Database flush optimization
        db.flush()
        db.commit()

        return {
            "status": "success",
            "inference_time_seconds": round(end_time - start_time, 4),
            "predictions": mapped
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ==========================================
# 🚀 ENDPOINT 2: REAL-TIME STREAMING API (FIXED)
# ==========================================
@router.post("/upload-stream")
async def upload_stream(
    file: UploadFile = File(...),
    window_size: int = Query(1),       # Dynamic window sizing
    delay: float = Query(0.05)         # Dynamic rate control
):
    """
    🔥 FIXED VERSION - Properly handles streaming without connection drops
    
    Key fixes:
    1. Write file to disk FIRST (avoid memory issues)
    2. Use proper async generator with yield
    3. Add proper error handling with try-except-finally
    4. Ensure stream is properly closed
    5. Don't set Content-Length for streaming responses
    """
    
    # Step 1: Write uploaded file to temporary location
    temp_file = None
    try:
        # Create temp file in system temp directory
        temp_fd, temp_file = tempfile.mkstemp(suffix=".csv")
        
        # Write file chunks to disk
        with os.fdopen(temp_fd, 'wb') as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)
        
        # Step 2: Read CSV and preprocess
        df = pd.read_csv(temp_file)

        if expected_columns:
            df = df.reindex(columns=expected_columns, fill_value=0)

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        data = df.values
        total_rows = len(data)

        # Step 3: Create proper async generator
        async def generate() -> AsyncGenerator[str, None]:
            """
            🔥 Proper streaming generator that:
            - Yields JSON lines (JSONL format)
            - Handles errors gracefully
            - Ensures proper stream termination
            """
            processed = 0
            errors = 0
            
            try:
                for i in range(0, len(data), window_size):
                    try:
                        print(f"🧪 Processing window: {i}/{total_rows}")
                        
                        window = data[i:i+window_size]

                        if len(window) == 0:
                            continue

                        # Calculate mean across window
                        row = np.mean(window, axis=0).reshape(1, -1)
                        
                        # 🔥 Soft-fail on feature mismatch
                        if scaler and hasattr(scaler, 'n_features_in_'):
                            if row.shape[1] != scaler.n_features_in_:
                                print(f"⚠️ Feature mismatch: Expected {scaler.n_features_in_}, got {row.shape[1]}")
                                errors += 1
                                continue

                        pred, conf = safe_predict(model, row, scaler)

                        result = {
                            "window_index": i // window_size,
                            "activity": label_map.get(int(pred[0]), "Unknown"),
                            "confidence": float(conf[0]),
                            "rows_in_window": len(window)
                        }

                        print("✅ Streaming result:", result)

                        # Yield as JSONL (JSON + newline)
                        yield json.dumps(result) + "\n"
                        processed += 1
                        
                        # Non-blocking delay to simulate real-time streaming
                        await asyncio.sleep(delay)
                        
                    except Exception as e:
                        print(f"⚠️ Window error at index {i}: {str(e)}")
                        errors += 1
                        # Send error result but continue processing
                        yield json.dumps({
                            "window_index": i // window_size,
                            "error": str(e),
                            "status": "skipped"
                        }) + "\n"
                        
                # Send final summary
                summary = {
                    "status": "complete",
                    "total_windows_processed": processed,
                    "total_errors": errors,
                    "total_rows": total_rows
                }
                yield json.dumps(summary) + "\n"
                print(f"✅ Stream completed: {processed} windows, {errors} errors")
                        
            except Exception as e:
                print(f"❌ STREAM ERROR: {str(e)}")
                # Send error message
                yield json.dumps({"error": str(e), "status": "failed"}) + "\n"
            
            finally:
                print("🔚 Stream generator finished")

        # Return streaming response with proper headers
        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",  # JSONL format
            headers={
                "Transfer-Encoding": "chunked",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        
        # Return error response (not streaming)
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }
    
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                print(f"🗑️ Cleaned up temp file: {temp_file}")
            except Exception as e:
                print(f"⚠️ Failed to clean temp file: {str(e)}")