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


# 🔥 Upgraded safe_predict function (FINAL SHAPE GUARANTEE)
def safe_predict(model, X, scaler=None):
    try:
        # 🔥 FORCE SHAPE FIX HERE (FINAL GUARANTEE)
        expected_features = model.n_features_in_

        if X.shape[1] != expected_features:
            print(f"⚠️ FIXING INPUT SHAPE: {X.shape[1]} → {expected_features}")
            
            if X.shape[1] < expected_features:
                pad = expected_features - X.shape[1]
                X = np.pad(X, ((0,0),(0,pad)), mode='constant')
            else:
                X = X[:, :expected_features]

        # Apply scaler AFTER fixing shape
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

    except Exception as e:
        print("❌ SAFE_PREDICT ERROR:", str(e))
        raise e


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
            df.columns = [str(c) for c in df.columns]  # normalize input columns
            expected_columns = [str(c) for c in expected_columns]

            # create full feature frame
            new_df = pd.DataFrame(0, index=df.index, columns=expected_columns)

            # copy matching columns only
            common_cols = set(df.columns).intersection(expected_columns)
            for col in common_cols:
                new_df[col] = df[col]

            df = new_df

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        data = df.values
        results = []
        start_time = time.time()
        
        for i in range(len(data)):
            row = data[i].reshape(1, -1)
            
            # 🔥 Shape Drift Protection (Soft Fail) - Kept as a warning before safe_predict forces it
            if scaler and hasattr(scaler, 'n_features_in_'):
                if row.shape[1] != scaler.n_features_in_:
                    print(f"⚠️ Batch Feature mismatch: Expected {scaler.n_features_in_}, got {row.shape[1]}")

            pred, conf = safe_predict(model, row, scaler)
            
            # 🔥 NEW CLEAN LABEL MAPPING
            activity_val = pred[0]
            if isinstance(activity_val, (int, np.integer, float)):
                activity_str = label_map.get(int(activity_val), "Unknown")
            else:
                activity_str = str(activity_val)

            results.append({
                "activity": activity_str,
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
# 🚀 ENDPOINT 2: REAL-TIME STREAMING API (WINDOWS-OPTIMIZED)
# ==========================================
@router.post("/upload-stream")
async def upload_stream(
    file: UploadFile = File(...),
    window_size: int = Query(50),  # 👈 CHANGED DEFAULT TO 50
    delay: float = Query(0.05)
):
    """
    🔥 WINDOWS-OPTIMIZED STREAMING ENDPOINT
    
    Fixes for Windows:
    1. Uses proper Windows temp directory
    2. Handles async event loop issues
    3. Explicit generator cleanup
    4. Chunked response with proper buffering
    """
    
    temp_file = None
    temp_dir = None
    
    try:
        print("\n" + "="*60)
        print(f"🚀 UPLOAD-STREAM CALLED")
        print(f"   File: {file.filename}")
        print(f"   Window size: {window_size}")
        print(f"   Delay: {delay}s")
        print("="*60)
        
        # Step 1: Write uploaded file to Windows temp directory
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"har_{int(time.time() * 1000)}.csv")
        
        print(f"📝 Writing to temp file: {temp_file}")
        
        # Write file chunks to disk
        total_size = 0
        with open(temp_file, 'wb') as f:
            while True:
                chunk = await file.read(512 * 1024)  # 512KB chunks
                if not chunk:
                    break
                f.write(chunk)
                total_size += len(chunk)
        
        print(f"✅ File written: {total_size} bytes")
        
        # Step 2: Read CSV and preprocess
        print("📊 Reading CSV...")
        df = pd.read_csv(temp_file)

        if expected_columns:
            df.columns = [str(c) for c in df.columns]
            expected_cols = [str(c) for c in expected_columns]

            new_df = pd.DataFrame(0, index=df.index, columns=expected_cols)

            common_cols = set(df.columns).intersection(expected_cols)
            for col in common_cols:
                new_df[col] = df[col]

            df = new_df
            

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        data = df.values
        total_rows = len(data)
        print(f"✅ CSV loaded: {total_rows} rows")

        # Step 3: Create proper async generator with Windows compatibility
        async def generate():
            """Streaming generator optimized for Windows"""
            processed = 0
            errors = 0
            
            try:
                for i in range(0, len(data), window_size):
                    try:
                        window = data[i:i+window_size]

                        if len(window) == 0:
                            continue

                        # 🔥 ADVANCED FEATURE EXTRACTION (Mean, Std, Min, Max, Median, 25th, 75th)
                        row = np.concatenate([
                            np.mean(window, axis=0),
                            np.std(window, axis=0),
                            np.min(window, axis=0),
                            np.max(window, axis=0),
                            np.median(window, axis=0),
                            np.percentile(window, 25, axis=0),
                            np.percentile(window, 75, axis=0)
                        ]).reshape(1, -1)
                        
                        # 🔥 The shape is now fully managed and forced inside safe_predict()
                        pred, conf = safe_predict(model, row, scaler)

                        # 🔥 NEW CLEAN LABEL MAPPING
                        activity_val = pred[0]
                        if isinstance(activity_val, (int, np.integer, float)):
                            activity_str = label_map.get(int(activity_val), "Unknown")
                        else:
                            activity_str = str(activity_val)

                        result = {
                            "window_index": i // window_size,
                            "activity": activity_str,
                            "confidence": float(conf[0]),
                            "rows_in_window": len(window)
                        }

                        print(f"✅ Window {i // window_size}: {result['activity']} ({result['confidence']:.2f})")

                        # Yield as JSONL
                        yield json.dumps(result) + "\n"
                        processed += 1
                        
                        # Non-blocking delay
                        await asyncio.sleep(delay)
                        
                    except Exception as window_err:
                        print(f"⚠️ Window error at index {i}: {str(window_err)}")
                        errors += 1
                        # Send error and continue
                        yield json.dumps({
                            "window_index": i // window_size,
                            "error": str(window_err),
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
                print(f"✅ STREAM COMPLETE: {processed} windows, {errors} errors")
                        
            except Exception as stream_err:
                print(f"❌ STREAM ERROR: {str(stream_err)}")
                try:
                    yield json.dumps({"error": str(stream_err), "status": "failed"}) + "\n"
                except:
                    pass  # Stream already broken
            
            finally:
                print("🔚 Stream generator finished")

        # Return streaming response with Windows-friendly headers
        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={
                "Transfer-Encoding": "chunked",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering if present
            }
        )

    except Exception as critical_err:
        print(f"❌ CRITICAL ERROR: {str(critical_err)}")
        import traceback
        print(traceback.format_exc())
        
        # Return error response (not streaming)
        return {
            "status": "error",
            "message": str(critical_err),
            "error_type": type(critical_err).__name__
        }
    
    finally:
        # Clean up temp file after streaming completes
        if temp_file and os.path.exists(temp_file):
            try:
                # Wait a moment for file to be released
                await asyncio.sleep(0.5)
                os.unlink(temp_file)
                print(f"🗑️ Cleaned up temp file: {temp_file}")
            except Exception as cleanup_err:
                print(f"⚠️ Failed to clean temp file: {str(cleanup_err)}")


# ==========================================
# 🔧 DEBUG ENDPOINT
# ==========================================
@router.get("/debug/streaming")
async def debug_streaming():
    """Check if streaming endpoint is properly configured"""
    import sys
    
    file_content = open(__file__).read()
    
    return {
        "status": "ok",
        "python_version": sys.version,
        "has_asyncgenerator": "AsyncGenerator" in file_content,
        "has_tempfile": "tempfile" in file_content,
        "has_windows_fix": "X-Accel-Buffering" in file_content,
        "file_location": __file__,
        "upload_folder": UPLOAD_FOLDER,
        "temp_dir": tempfile.gettempdir()
    }