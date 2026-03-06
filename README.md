# 🛒 Haalkhata Engine — Deployment Guide

## Architecture

```
Phone / Browser  ──►  index.html (Vercel)
                            │
                            │  POST /detect-base64
                            ▼
                  FastAPI Backend (Render)
                            │
                   YOLOv5 + EasyOCR
```

---

## Step 1 — Deploy Backend to Render

### 1a. Push to GitHub
```bash
git init
git add .
git commit -m "Haalkhata Engine backend"
git remote add origin https://github.com/YOUR_USERNAME/haalkhata-backend.git
git push -u origin main
```

### 1b. Create Render Web Service
1. Go to https://render.com → **New → Web Service**
2. Connect your GitHub repo
3. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Region**: Singapore (closest to Bangladesh)
   - **Plan**: Free (sleeps after 15min) or Starter $7/mo (always on)
4. Click **Deploy**
5. Wait ~5-10 min for first deploy (downloads YOLO weights ~14MB + EasyOCR models)
6. Your URL will be: `https://haalkhata-engine-api.onrender.com`

### Test backend
```bash
curl https://your-app.onrender.com/health
# → {"status":"healthy"}
```

---

## Step 2 — Deploy Frontend to Vercel

### Option A — Drag & Drop (easiest)
1. Go to https://vercel.com → **Add New Project**
2. Drag the folder containing `index.html` onto the page
3. Done — get your URL like `https://haalkhata.vercel.app`

### Option B — GitHub
```bash
# in a separate folder with just index.html
git init && git add index.html && git commit -m "frontend"
git remote add origin https://github.com/YOUR_USERNAME/haalkhata-frontend.git
git push -u origin main
# then import on Vercel
```

---

## Step 3 — Connect Frontend to Backend

1. Open your Vercel app on your phone
2. Paste your Render URL into the API field at the top
3. Tap **Test** — should show "Connected ✓"
4. URL is saved in localStorage for next visits

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/detect` | Upload image file, get detections |
| POST | `/detect-base64` | Send `{"image": "<base64>"}`, get detections |

### Response format
```json
{
  "count": 2,
  "detections": [
    {
      "label": "bottle",
      "confidence": 0.91,
      "bbox": [30, 50, 150, 200],
      "ocr_text": "Mineral Water 500ml"
    }
  ],
  "annotated": "<base64 JPEG with bounding boxes drawn>"
}
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Render first request is slow | Free tier cold starts take 30-60s. Upgrade to Starter for always-on. |
| CORS error | Already handled — all origins allowed in `main.py` |
| Camera not working on phone | Must be served over HTTPS. Vercel gives HTTPS automatically. |
| Voice not working on iPhone | Use Chrome on Android. Safari blocks SpeechRecognition API. |
| OCR returning empty | EasyOCR needs clear text. Small/blurry labels may return empty string. |

---

## File Structure
```
haalkhata-backend/
├── main.py           ← FastAPI app (YOLO + EasyOCR)
├── requirements.txt  ← Python dependencies
├── render.yaml       ← Render deployment config
├── Procfile          ← Railway fallback
├── index.html        ← Frontend (deploy separately to Vercel)
└── README.md         ← This file
```
