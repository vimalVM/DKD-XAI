# Deployment Guide: Render (Backend) + Netlify (Frontend)

This guide walks you through deploying the **DKD-XAI** platform to production using **Render** for the Python FastAPI backend and **Netlify** for the React Vite frontend.

---

## Project Structure Overview

The repository is organized into distinct `backend/` and `frontend/` directories:

```
DKD-XAI_Updated/
├── backend/                       <-- Deploy to RENDER
│   ├── backend_server.py          # FastAPI application & REST endpoints
│   ├── dkd_ai_agent.py            # SHAP & Groq clinical reasoning agent
│   ├── dkd_model_pipeline.py      # Feature definitions & PyTorch MLP model
│   ├── dkd_evidence_base.json     # ADA 2026 / KDIGO curated guidelines
│   ├── dkd_nhanes_final.csv       # NHANES cohort reference for patient lookups
│   ├── results/                   # Trained models (best_model.joblib for each task)
│   ├── requirements.txt           # Python dependencies for Render
│   ├── Procfile                   # Process start command for Render
│   ├── runtime.txt                # Python 3.11.9
│   └── .env.example               # Template for GROQ_API_KEY
│
├── frontend/                      <-- Deploy to NETLIFY
│   ├── src/                       # React components & UI
│   ├── public/
│   │   └── _redirects             # Netlify SPA catch-all rule
│   ├── package.json               # Node.js dependencies
│   ├── vite.config.js             # Vite configuration with local proxy
│   ├── netlify.toml               # Netlify build & redirect rules
│   └── .env.example               # Template for VITE_API_URL
│
├── render.yaml                    # Optional 1-click Render blueprint
└── netlify.toml                   # Root Netlify configuration pointing to frontend
```

---

## Step 1: Push Code to GitHub

Make sure your repository has your latest commits and is pushed to GitHub:

```bash
git add .
git commit -m "Organize project into backend and frontend for Render and Netlify"
git push origin main
```

---

## Step 2: Deploy Backend on Render

Render will host the FastAPI server with PyTorch, XGBoost, SHAP, and Groq integration.

1. Go to [https://dashboard.render.com](https://dashboard.render.com) and log in.
2. Click **New +** > **Web Service**.
3. Connect your GitHub repository (`DKD-XAI`).
4. Configure the Web Service settings:
   - **Name:** `dkd-xai-backend` (or any unique name)
   - **Region:** Choose the region closest to you (e.g., Oregon, Frankfurt, Singapore)
   - **Branch:** `main`
   - **Root Directory:** `backend`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend_server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free` (or Starter for faster cold starts)
5. Under **Environment Variables**, add:
   - `PYTHON_VERSION`: `3.11.9`
   - `GROQ_API_KEY`: *(Your Groq API key from console.groq.com)*
6. Click **Create Web Service**.
7. Wait 2–3 minutes for the build to finish. Once live, Render gives you a public URL (e.g. `https://dkd-xai-backend.onrender.com`).
8. **Verify:** Open `https://your-backend.onrender.com/api/health` in your browser. You should see:
   ```json
   {
     "status": "online",
     "loaded_models": ["dkd_any", "early_dkd"],
     "has_groq_key": true,
     "sample_cohort_size": 4024
   }
   ```

---

## Step 3: Deploy Frontend on Netlify

Netlify will build and host the fast React Single Page Application (SPA).

1. Go to [https://app.netlify.com](https://app.netlify.com) and log in.
2. Click **Add new site** > **Import an existing project**.
3. Select **GitHub** and authorize your `DKD-XAI` repository.
4. Configure the Build settings:
   - **Base directory:** `frontend`
   - **Build command:** `npm run build`
   - **Publish directory:** `frontend/dist` (or `dist` if base directory is set to `frontend`)
5. Under **Environment variables**, click **Add a variable**:
   - **Key:** `VITE_API_URL`
   - **Value:** `https://your-backend.onrender.com` *(Use the exact URL from Step 2, without a trailing slash)*
6. Click **Deploy site**.
7. In ~1 minute, Netlify will publish your site (e.g. `https://dkd-xai.netlify.app`).

---

## Step 4: Verify Full Integration

1. Open your Netlify site URL in your browser.
2. The header should show **System Online** with a green pulse dot.
3. Select any benchmark patient from the dropdown (e.g., *Sample 1: High Risk Case*).
4. Verify:
   - The **Risk Meter** updates with live probability percentages.
   - The **SHAP Factor Breakdown** shows positive (risk-increasing) and negative (protective) bars.
   - The **Conversational Assistant** generates evidence-grounded reports with verified ADA 2026 citations.

---

## Local Development (Optional)

If you wish to run the project locally on your computer:

### 1. Run Backend:
```bash
cd backend
python -m uvicorn backend_server:app --reload --port 8000
```
Runs at `http://127.0.0.1:8000`.

### 2. Run Frontend:
```bash
cd frontend
npm install
npm run dev
```
Runs at `http://localhost:5173`. In development mode, Vite automatically proxies `/api/*` to `http://127.0.0.1:8000`.
