import os
import json
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Load scalers & metadata ────────────────────────────────────────────────────
print("Loading scalers...")
with open(os.path.join(MODEL_DIR, "all_scalers.pkl"), "rb") as f:
    all_scalers = pickle.load(f)

print("Loading sensor metadata...")
with open(os.path.join(MODEL_DIR, "sensor_metadata.json"), "r") as f:
    sensor_metadata = json.load(f)

sensor_ids = list(sensor_metadata.keys())
print(f"Loaded {len(sensor_ids)} sensors")

# ── Load Random Forest ─────────────────────────────────────────────────────────
print("Loading Random Forest model...")
with open(os.path.join(MODEL_DIR, "random_forest_best.pkl"), "rb") as f:
    rf_model = pickle.load(f)
print("RF model loaded ✓")

# ── Load LSTM (version-safe) ───────────────────────────────────────────────────
lstm_model = None

def load_lstm_safe():
    """Try multiple strategies to load the LSTM across Keras versions."""
    global lstm_model

    # Strategy 1: native .keras format (TF 2.12+)
    keras_path = os.path.join(MODEL_DIR, "lstm_best.keras")
    if os.path.exists(keras_path):
        try:
            from tensorflow.keras.models import load_model
            lstm_model = load_model(keras_path, compile=False)
            print("LSTM loaded from .keras format ✓")
            return
        except Exception as e:
            print(f"Strategy 1 (.keras) failed: {e}")

    # Strategy 2: legacy .h5 with custom InputLayer patch
    h5_path = os.path.join(MODEL_DIR, "lstm_best.h5")
    if os.path.exists(h5_path):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras.layers import InputLayer

            class CompatInputLayer(InputLayer):
                def __init__(self, **kwargs):
                    kwargs.pop("batch_shape", None)
                    kwargs.pop("optional", None)
                    super().__init__(**kwargs)

            lstm_model = load_model(
                h5_path,
                compile=False,
                custom_objects={"InputLayer": CompatInputLayer}
            )
            print("LSTM loaded from .h5 with compat patch ✓")
            return
        except Exception as e:
            print(f"Strategy 2 (.h5 compat) failed: {e}")

    # Strategy 3: rebuild architecture and load weights only
    h5_path = os.path.join(MODEL_DIR, "lstm_best.h5")
    if os.path.exists(h5_path):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout

            model = Sequential([
                LSTM(64, input_shape=(12, 6), return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(1)
            ])
            model.load_weights(h5_path, by_name=False, skip_mismatch=False)
            lstm_model = model
            print("LSTM rebuilt + weights loaded ✓")
            return
        except Exception as e:
            print(f"Strategy 3 (rebuild) failed: {e}")

    print("WARNING: LSTM could not be loaded — only RF predictions available")

print("Loading LSTM model...")
load_lstm_safe()

# ── Helper: time encodings ─────────────────────────────────────────────────────
def get_time_features(step_index, steps_per_day=288):
    angle = 2 * np.pi * step_index / steps_per_day
    return np.sin(angle), np.cos(angle)

# ── Helper: traffic condition ──────────────────────────────────────────────────
def classify_speed(speed_mph):
    if speed_mph < 30:
        return "Congested",   "danger"
    elif speed_mph < 50:
        return "Moderate",    "warning"
    else:
        return "Free Flow",   "success"

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/sensors")
def get_sensors():
    return jsonify(sensor_metadata)

@app.route("/api/models")
def get_models():
    return jsonify({
        "models": ["LSTM", "Random Forest"],
        "lstm_available": lstm_model is not None
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data        = request.get_json()
        sensor_id   = str(data.get("sensor_id"))
        model_type  = data.get("model", "LSTM")
        speeds      = data.get("speeds", [])      # list of 12 raw mph values
        time_step   = int(data.get("time_step", 0))

        if len(speeds) != 12:
            return jsonify({"error": "Exactly 12 speed readings required"}), 400

        if sensor_id not in all_scalers:
            return jsonify({"error": f"Unknown sensor: {sensor_id}"}), 400

        scaler = all_scalers[sensor_id]

        # Normalise speeds
        speeds_arr    = np.array(speeds, dtype=np.float32).reshape(-1, 1)
        speeds_scaled = scaler.transform(speeds_arr).flatten()

        # Build time features
        time_features = np.array([
            get_time_features(time_step + i) for i in range(12)
        ])  # shape (12, 2)

        # Build feature matrix (12, 6): speed + 4 dummy cols + 2 time cols
        # Matches training: [speed, sin_hour, cos_hour, sin_dow, cos_dow, is_weekend]
        dummy = np.zeros((12, 3))
        features = np.column_stack([
            speeds_scaled,
            time_features[:, 0],   # sin_time
            time_features[:, 1],   # cos_time
            dummy
        ])  # shape (12, 6)

        # ── LSTM prediction ──
        if model_type == "LSTM":
            if lstm_model is None:
                return jsonify({"error": "LSTM model not available"}), 503
            X = features.reshape(1, 12, 6)
            pred_scaled = float(lstm_model.predict(X, verbose=0)[0][0])

        # ── RF prediction ──
        else:
            X = features.reshape(1, -1)   # (1, 72)
            pred_scaled = float(rf_model.predict(X)[0])

        # Inverse-transform
        pred_speed = float(scaler.inverse_transform([[pred_scaled]])[0][0])
        pred_speed = max(0.0, min(80.0, pred_speed))

        condition, badge = classify_speed(pred_speed)

        return jsonify({
            "sensor_id":   sensor_id,
            "model":       model_type,
            "predicted_speed": round(pred_speed, 2),
            "condition":   condition,
            "badge":       badge,
            "unit":        "mph"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
