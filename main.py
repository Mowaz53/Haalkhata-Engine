from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
import io
import os
import logging
import httpx
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Haalkhata Engine API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-load heavy models so cold start is fast ──────────────────────────────
_yolo = None
_ocr  = None

def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        logger.info("Loading YOLOv5s model...")
        _yolo = YOLO("yolov5su.pt")
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
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        ocr   = det.get("ocr_text", "")
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
    return {"status": "ok", "service": "Haalkhata Engine API", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}


# ── YOLO + OCR detection ──────────────────────────────────────────────────────
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = decode_image(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image: {e}")

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


@app.post("/detect-base64")
async def detect_base64(payload: dict):
    try:
        b64 = payload.get("image", "")
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


# ── GEMINI AGENT endpoint ─────────────────────────────────────────────────────
# Gemini key is stored as GEMINI_API_KEY environment variable on Render
# Never hardcoded — set it in Render dashboard → Environment

@app.post("/gemini-detect")
async def gemini_detect(payload: dict):
    """
    Accepts { "image": "<base64 string>" }
    Forwards to Gemini Vision API using server-side key
    Returns structured product identification results
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise HTTPException(status_code=503, detail="Gemini API key not configured on server")

    b64 = payload.get("image", "")
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    if not b64:
        raise HTTPException(status_code=400, detail="No image provided")

    prompt = """You are a retail product identification assistant helping a visually impaired shopkeeper in Bangladesh.

Analyse this image and identify ALL visible objects and products.

For each item respond in this exact JSON format:
{
  "items": [
    {
      "name": "product name",
      "brand": "brand if visible or Unknown",
      "label_text": "any text visible on the product",
      "category": "category e.g. Beverage, Grocery, Personal Care",
      "confidence": "High / Medium / Low",
      "notes": "brief useful note for a blind shopkeeper"
    }
  ],
  "scene_summary": "one sentence describing the overall scene"
}

Be specific with product names. If you see Bengali text, include it. Respond ONLY with valid JSON."""

    gemini_payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
            ]
        }],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024}
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}",
                json=gemini_payload
            )
        if resp.status_code != 200:
            err = resp.json()
            msg = err.get("error", {}).get("message", f"HTTP {resp.status_code}")
            raise HTTPException(status_code=502, detail=f"Gemini error: {msg}")

        data = resp.json()
        raw  = data["candidates"][0]["content"]["parts"][0]["text"]
        return JSONResponse({"raw": raw, "status": "ok"})

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Gemini request timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

# ── PaddleOCR ─────────────────────────────────────────────────────────────────
_paddle = None
def get_paddle():
    global _paddle
    if _paddle is None:
        from paddleocr import PaddleOCR
        logger.info("Loading PaddleOCR...")
        _paddle = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        logger.info("PaddleOCR ready.")
    return _paddle

@app.post("/paddle-ocr")
async def paddle_ocr(payload: dict):
    try:
        image_data = payload.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        ocr = get_paddle()
        result = ocr.ocr(img, cls=True)

        lines = []
        if result and result[0]:
            for line in result[0]:
                text  = line[1][0]
                score = line[1][1]
                if score > 0.5 and text.strip():
                    lines.append(text.strip())

        return JSONResponse({"text": "\n".join(lines), "lines": lines, "status": "ok"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PaddleOCR error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR error: {str(e)}")
