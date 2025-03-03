from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn

app = FastAPI()

# --- Request & Response Models ---
class InferenceRequest(BaseModel):
    image: str  # base64-encoded image string

class DetectionResult(BaseModel):
    object: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

class InferenceResponse(BaseModel):
    annotated_image: str  # base64-encoded annotated image
    detections: List[DetectionResult]

# --- Load Models ---
try:
    # Path to gate model (adjust path if needed)
    gate_model = YOLO("models/gate-newest.engine")
    gate_model.conf = 0.4
    gate_model.iou = 0.5
    gate_classes = ["Gate"]
except Exception as e:
    print("Error loading gate model:", e)
    gate_model = None

try:
    # Path to bucket model (adjust path if needed)
    bucket_model = YOLO("models/bucket.engine")
    bucket_model.conf = 0.25
    bucket_model.iou = 0.5
    bucket_classes = ["Blue Bucket", "Red Bucket"]
except Exception as e:
    print("Error loading bucket model:", e)
    bucket_model = None

# --- Utility Functions ---
def decode_image(image_base64: str):
    image_bytes = base64.b64decode(image_base64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def encode_image(img):
    ret, buffer = cv2.imencode('.jpg', img)
    if not ret:
        raise ValueError("Could not encode image")
    return base64.b64encode(buffer).decode('utf-8')

# --- Endpoints ---
@app.post("/inference/gate", response_model=InferenceResponse)
def inference_gate(request: InferenceRequest):
    if gate_model is None:
        raise HTTPException(status_code=500, detail="Gate model not loaded")
    try:
        img = decode_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {e}")

    try:
        results = gate_model.predict(
            source=img,
            conf=gate_model.conf,
            iou=gate_model.iou,
            device=0,
            half=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    annotated_image = img.copy()
    detections = []
    for result in results:
        if not hasattr(result, 'boxes'):
            continue
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax]
            conf = float(box.conf[0].cpu().numpy())
            detections.append({
                "object": gate_classes[0],
                "confidence": conf,
                "x1": int(xyxy[0]),
                "y1": int(xyxy[1]),
                "x2": int(xyxy[2]),
                "y2": int(xyxy[3])
            })
            cv2.rectangle(annotated_image,
                          (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])),
                          (255, 0, 0), 2)
            label = f"{gate_classes[0]}: {conf:.2f}"
            cv2.putText(annotated_image, label, (int(xyxy[0]), int(xyxy[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    try:
        annotated_image_b64 = encode_image(annotated_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image encoding error: {e}")

    return InferenceResponse(annotated_image=annotated_image_b64, detections=detections)

@app.post("/inference/bucket", response_model=InferenceResponse)
def inference_bucket(request: InferenceRequest):
    if bucket_model is None:
        raise HTTPException(status_code=500, detail="Bucket model not loaded")
    try:
        img = decode_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {e}")

    try:
        results = bucket_model.predict(
            source=img,
            conf=bucket_model.conf,
            iou=bucket_model.iou,
            device=0,
            half=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    annotated_image = img.copy()
    detections = []
    for result in results:
        if not hasattr(result, 'boxes'):
            continue
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax]
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            if cls_id < 0 or cls_id >= len(bucket_classes):
                continue
            detections.append({
                "object": bucket_classes[cls_id],
                "confidence": conf,
                "x1": int(xyxy[0]),
                "y1": int(xyxy[1]),
                "x2": int(xyxy[2]),
                "y2": int(xyxy[3])
            })
            # Draw detection bounding box
            cv2.rectangle(annotated_image,
                          (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])),
                          (0, 255, 0), 2)
            label = f"{bucket_classes[cls_id]}: {conf:.2f}"
            cv2.putText(annotated_image, label, (int(xyxy[0]), int(xyxy[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    try:
        annotated_image_b64 = encode_image(annotated_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image encoding error: {e}")

    return InferenceResponse(annotated_image=annotated_image_b64, detections=detections)

if __name__ == "__main__":
    # Run the server on all interfaces on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
