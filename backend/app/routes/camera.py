"""
Camera-Based Human Activity Recognition Backend
FastAPI router with WebSocket support for real-time pose analysis
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import base64
import numpy as np
import cv2
import asyncio
import time
import logging
from typing import Optional
import traceback
from collections import deque

from app.services.pose_estimator import PoseEstimator
from app.services.activity_classifier import ActivityClassifier
from app.services.sequence_predictor import SequencePredictor
from app.models.schemas import PredictionResult, FrameAnalysis

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App Init ────────────────────────────────────────────────────────────────
router = APIRouter()

# ─── Shared ML Components (loaded once at startup) ───────────────────────────
pose_estimator: Optional[PoseEstimator] = None
activity_classifier: Optional[ActivityClassifier] = None
sequence_predictor: Optional[SequencePredictor] = None

# ─── Shared Sensor State ─────────────────────────────────────────────────────
class SharedSensorState:
    def __init__(self):
        self.activity = "Walking" # 🔥 Placeholder until real data flows
        self.confidence = 0.85
        self.last_update = 0.0

# Instantiate the global state for multimodal fusion
sensor_state = SharedSensorState()


@router.on_event("startup")
async def startup_event():
    global pose_estimator, activity_classifier, sequence_predictor
    logger.info("Loading ML components...")
    pose_estimator = PoseEstimator()
    activity_classifier = ActivityClassifier()
    sequence_predictor = SequencePredictor()
    logger.info("✅ All ML components loaded.")


# ─── REST Endpoints ──────────────────────────────────────────────────────────

@router.get("/camera/health")
async def health():
    return {"status": "ok", "components": {
        "pose_estimator": pose_estimator is not None,
        "activity_classifier": activity_classifier is not None,
        "sequence_predictor": sequence_predictor is not None,
    }}


# 🔥 Webhook for external Sensor/IMU scripts to push live data
@router.post("/sensor/update")
async def update_sensor(payload: dict):
    sensor_state.activity = payload.get("activity", "Unknown")
    sensor_state.confidence = float(payload.get("confidence", 0.0))
    sensor_state.last_update = time.time()
    return {"status": "success"}


# 🔥 The Upgraded Real-Time Camera Stream Endpoint 🔥
@router.get("/camera-stream")
def real_camera_stream():
    def generate():
        cap = cv2.VideoCapture(0)
        
        # Initialize State Trackers per connection
        activity_history = deque(maxlen=5)  
        conf_history = deque(maxlen=5)      
        timeline = deque(maxlen=20)         
        
        frame_id = 0 
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Could not read from camera.")
                    break

                frame_id += 1

                # Get raw predictions
                result = _process_frame(frame)

                # 🧠 Activity Smoothing (Mode)
                activity_history.append(result.activity)
                final_activity = max(set(activity_history), key=activity_history.count)

                # 🚀 Confidence Smoothing (Moving Average)
                conf_history.append(result.confidence)
                avg_conf = sum(conf_history) / len(conf_history)

                # 🚀 Add Activity Timeline
                timeline.append(final_activity)

                # Fetch live sensor data
                sensor_pred = sensor_state.activity
                sensor_conf = sensor_state.confidence

                # Failsafe: Mark offline if no updates in 2 seconds
                if time.time() - sensor_state.last_update > 2.0:
                    sensor_pred = "Offline"
                    sensor_conf = 0.0

                # 🔥 STEP 1 & 3: Smarter Weighted Fusion
                camera_weight = 0.6
                sensor_weight = 0.4

                if sensor_pred == final_activity:
                    # Agreement → Blend confidence
                    final_pred = final_activity
                    final_conf = (camera_weight * avg_conf) + (sensor_weight * sensor_conf)
                else:
                    # Disagreement → Weighted Trust
                    if (avg_conf * camera_weight) > (sensor_conf * sensor_weight):
                        final_pred = final_activity
                    else:
                        final_pred = sensor_pred
                    final_conf = max(avg_conf, sensor_conf)

                # 🚀 OPTIONAL PRO FEATURE: Disagreement flag
                is_disagreement = (final_activity != sensor_pred) and (sensor_pred != "Offline")

                # 🔥 STEP 2: The Upgraded Ultimate Payload
                payload = {
                    "frame_id": frame_id,                     
                    "timestamp": time.time(),
                    "final_activity": final_pred,
                    "final_conf": round(final_conf, 4),
                    "camera": final_activity,
                    "camera_conf": round(avg_conf, 4),
                    "sensor": sensor_pred,
                    "sensor_conf": round(sensor_conf, 4),
                    "history": list(timeline),
                    "disagreement": is_disagreement 
                }
                
                yield json.dumps(payload) + "\n"

                time.sleep(0.1)
        finally:
            logger.info("Releasing camera hardware.")
            cap.release()

    return StreamingResponse(generate(), media_type="text/plain")


@router.post("/predict/frame")
async def predict_frame(payload: dict):
    """
    Single-frame prediction via HTTP POST.
    Body: { "image": "<base64-encoded JPEG>" }
    """
    try:
        b64 = payload.get("image", "")
        if not b64:
            raise HTTPException(status_code=400, detail="Missing 'image' field")

        if "," in b64:
            b64 = b64.split(",", 1)[1]

        img_bytes = base64.b64decode(b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        result = _process_frame(frame)
        return JSONResponse(content=result.dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ─── WebSocket Endpoint ───────────────────────────────────────────────────────

@router.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """
    Real-time camera stream via WebSocket.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")

    local_seq = SequencePredictor()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "frame":
                continue

            b64 = msg.get("image", "")
            if not b64:
                continue

            if "," in b64:
                b64 = b64.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"error": "Could not decode frame"})
                    continue

                result = _process_frame(frame, local_seq)
                await websocket.send_json(result.dict())

            except Exception as e:
                logger.warning(f"Frame processing error: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ─── Internal Processing ──────────────────────────────────────────────────────

def _process_frame(
    frame: np.ndarray,
    seq_predictor: Optional[SequencePredictor] = None
) -> "PredictionResult":
    """
    Full pipeline:
      frame → pose landmarks → features → activity + confidence → next action
    """
    sp = seq_predictor or sequence_predictor

    # 1. Pose estimation
    landmarks, annotated_frame = pose_estimator.process(frame)

    if landmarks is None:
        return PredictionResult(
            activity="No person detected",
            confidence=0.0,
            next_action="Unknown",
            landmarks=[],
            history=sp.get_history(),
            pose_detected=False
        )

    # 2. Feature engineering
    features = pose_estimator.landmarks_to_features(landmarks)

    # 3. Activity classification
    activity, confidence, all_probs = activity_classifier.predict(features)

    # 4. Update sequence memory + next-action prediction
    next_action = sp.update_and_predict(activity)

    # 5. Serialize landmarks for frontend
    lm_list = [
        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
        for lm in landmarks
    ]

    return PredictionResult(
        activity=activity,
        confidence=round(float(confidence), 4),
        next_action=next_action,
        landmarks=lm_list,
        history=sp.get_history(),
        pose_detected=True,
        all_probabilities=all_probs
    )