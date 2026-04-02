# Traffic Forecasting Web App — Deployment Guide

## Project Structure
webapp/
├── app.py                    # Flask backend
├── requirements.txt          # Python dependencies
├── models/
│   ├── lstm_best.h5          # LSTM_Large model
│   ├── random_forest_best.pkl# Random Forest model
│   ├── all_scalers.pkl       # Per-sensor MinMaxScalers
│   └── sensor_metadata.json  # 207 sensor GPS coordinates
├── templates/
│   └── index.html            # Frontend (map + chart + prediction UI)
└── static/                   # (empty — CSS/JS loaded from CDN)

## API Endpoints
GET  /                  → serves the web frontend
GET  /api/sensors       → returns all 207 sensor IDs + GPS coordinates
GET  /api/models        → returns available models + MAE/RMSE metrics
POST /api/predict       → predicts traffic speed

POST /api/predict — Request body (JSON):
{
  "sensor_id": "773869",
  "model": "lstm",          // "lstm" or "rf"
  "speeds": [65,63,61,58,55,52,49,46,44,42,40,38]  // 12 values in mph
}

POST /api/predict — Response:
{
  "sensor_id": "773869",
  "model": "lstm",
  "predicted_speed_mph": 36.5,
  "condition": "Moderate",
  "color": "#F39C12",
  "lat": 34.15497,
  "lng": -118.31829,
  "input_speeds": [...]
}

## Option 1: Run Locally
pip install -r requirements.txt
cd webapp
python app.py
# Open http://localhost:5000

## Option 2: Deploy on Render (Free )
1. Upload the webapp/ folder to a GitHub repository
2. Go to render.com → New Web Service
3. Connect your GitHub repo
4. Set:
   - Build Command: pip install -r requirements.txt
   - Start Command: gunicorn app:app
5. Deploy — Render gives a free public URL

## Option 3: Deploy on Railway (Free)
1. Upload webapp/ to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Set start command: gunicorn app:app
4. Deploy

## Option 4: Deploy on PythonAnywhere (Free)
1. Upload all files to PythonAnywhere
2. Set WSGI file to point to app.py
3. Install requirements in the console

## Notes
- lstm_best.h5 is ~1.5 MB — loads in ~3 seconds on free tier servers
- random_forest_best.pkl is ~50 MB — may take 10-15 seconds to load on free tier
- For faster startup, consider using only the LSTM model in production