# Waste Classifier — Deployment Package

**Live demo:** [abxwastemanagment.netlify.app](https://abxwastemanagment.netlify.app/)

> Note: the frontend above is permanently live, but the backend it talks to
> currently runs via `ngrok` on a local machine (see "Current setup" below).
> The demo only works while that machine + ngrok tunnel are running. For a
> backend that's live 24/7, move it to Render or Hugging Face Spaces (Section 2).

A 4-class waste image classifier (Hazardous / Non-Recyclable / Organic / Recyclable)
built from `classification.ipynb`, packaged as:

- `backend/` — FastAPI service that loads `waste.pth` and exposes `POST /predict`
- `frontend/` — a single static `index.html` (no build step) with drag-and-drop upload

## Current setup (as deployed right now)

- **Frontend**: deployed on Netlify at the live demo link above.
- **Backend**: running locally with `uvicorn`, exposed to the internet with
  `ngrok http 8000`. The ngrok URL changes every time the tunnel restarts —
  update `API_URL` in `frontend/index.html` and redeploy to Netlify whenever
  that happens.

## 1. Run it locally first

**Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```
Visit `http://localhost:8000` — you should see `{"status": "ok", ...}`.

**Frontend**
Just open `frontend/index.html` in your browser (double-click it, or use the
VS Code "Live Server" extension). `API_URL` inside the `<script>` tag is
already set to `http://localhost:8000`, so it will talk to
