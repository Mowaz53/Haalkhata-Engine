from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
import io
import os
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Haalkhata Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-load heavy models so cold start is fast ──────────────────────────────
_yolo  = None
_ocr   = None

def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        logger.info("Loading YOLOv5s model...")
        _yolo = YOLO("yolov5su.pt")   # auto-downloads on first run
        logger.info("YOLO ready.")
    return _yolo

def get_ocr():
    global _ocr
    if _ocr is None:
        import easyocr
        logger.info("Loading EasyOCR...")
        _ocr = easyocr.Reader(['en'], gpu=False)
        logger.info("EasyOCR ready.")
    return _ocr

# ── helpers ───────────────────────────────────────────────────────────────────
def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def run_ocr_on_region(img_region: np.ndarray) -> str:
    if img_region.size == 0:
        return ""
    try:
        results = get_ocr().readtext(img_region, detail=0)
        return " ".join(results).strip() if results else ""
    except Exception as e:
        logger.warning(f"OCR error: {e}")
        return ""

def draw_boxes(img: np.ndarray, detections: list) -> str:
    """Draw bounding boxes and return base64 annotated image."""
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        ocr   = det["ocr_text"]
        conf  = det["confidence"]

        cv2.rectangle(out, (x1,y1), (x2,y2), (0,245,170), 2)

        tag = f"{label}"
        if ocr: tag += f" | {ocr[:20]}"
        tag_w = len(tag) * 8 + 10
        cv2.rectangle(out, (x1, y1-24), (x1+tag_w, y1), (0,245,170), -1)
        cv2.putText(out, tag, (x1+5, y1-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buf).decode("utf-8")


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "Haalkhata Engine API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Accepts a JPEG/PNG image, runs YOLOv5 + EasyOCR,
    returns detections + annotated image.
    """
    try:
        raw = await file.read()
        img = decode_image(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image: {e}")

    # ── YOLO inference ──────────────────────────────────────────────────────
    try:
        results = get_yolo()(img, conf=0.35, iou=0.5, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO error: {e}")

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label      = result.names[int(box.cls[0])]
            confidence = round(float(box.conf[0]), 3)

            # ── OCR on cropped region ───────────────────────────────────────
            region   = img[max(0,y1):y2, max(0,x1):x2]
            ocr_text = run_ocr_on_region(region)

            detections.append({
                "label":      label,
                "confidence": confidence,
                "bbox":       [x1, y1, x2, y2],
                "ocr_text":   ocr_text,
            })

    # ── annotated image ─────────────────────────────────────────────────────
    annotated_b64 = draw_boxes(img, detections)

    return JSONResponse({
        "count":      len(detections),
        "detections": detections,
        "annotated":  annotated_b64,   # base64 JPEG
    })


@app.post("/detect-base64")
async def detect_base64(payload: dict):
    """
    Alternative endpoint — accepts { "image": "<base64 string>" }
    Useful for browsers that send canvas data directly.
    """
    try:
        b64 = payload.get("image", "")
        # strip data URI prefix if present
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        img = decode_image(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad base64 image: {e}")

    try:
        results = get_yolo()(img, conf=0.35, iou=0.5, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO error: {e}")

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label      = result.names[int(box.cls[0])]
            confidence = round(float(box.conf[0]), 3)
            region     = img[max(0,y1):y2, max(0,x1):x2]
            ocr_text   = run_ocr_on_region(region)
            detections.append({
                "label":      label,
                "confidence": confidence,
                "bbox":       [x1, y1, x2, y2],
                "ocr_text":   ocr_text,
            })

    annotated_b64 = draw_boxes(img, detections)

    return JSONResponse({
        "count":      len(detections),
        "detections": detections,
        "annotated":  annotated_b64,
    })
