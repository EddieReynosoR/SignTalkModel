from flask import Flask, request
from flask_socketio import SocketIO, emit
import numpy as np
import mediapipe as mp
import cv2
import base64
import os
from pathlib import Path
import tensorflow as tf
import logging
from collections import deque
import threading
import time

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=5 * 1024 * 1024,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sign-ws")

# -------------------------
# Utils
# -------------------------
def extract_key_points(results) -> np.ndarray:
    pose = (
        np.array(
            [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4, dtype=np.float32)
    )

    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )

    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )

    return np.concatenate([pose, lh, rh]).astype(np.float32)

def decode_data_url_to_frame(frame_data: str):
    """
    Accepts:
      - data URL: "data:image/jpeg;base64,....."
      - raw base64 string
    Returns:
      - frame (BGR) or None
      - error message or None
    """
    if not isinstance(frame_data, str) or not frame_data:
        return None, "Empty frame payload"

    try:
        if "," in frame_data:
            _, b64 = frame_data.split(",", 1)
        else:
            b64 = frame_data

        # Remove whitespace/newlines if any
        b64 = b64.strip()

        image_data = base64.b64decode(b64, validate=False)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, "Could not decode frame"
        return frame, None
    except Exception as e:
        return None, f"Decode error: {e}"

# -------------------------
# Load actions and SavedModel (production)
# -------------------------
base_dir = Path(__file__).resolve().parent
DATA_PATH = base_dir / "MP_Data"

actions = np.array(
        sorted([f for f in os.listdir(DATA_PATH) if (DATA_PATH / f).is_dir()]),
        dtype=str)

saved_model = tf.saved_model.load(str(base_dir / "sign_model"))

signature_name = "serving_default" if "serving_default" in saved_model.signatures else next(iter(saved_model.signatures))
infer = saved_model.signatures[signature_name]

input_key = next(iter(infer.structured_input_signature[1].keys()))
output_key = next(iter(infer.structured_outputs.keys()))

logger.info("SavedModel signature=%s, input_key=%s, output_key=%s", signature_name, input_key, output_key)

FEATURES = 33 * 4 + 21 * 3 + 21 * 3
SEQ_LEN = 30

def make_prediction(sequence_data: deque):
    """Prediction using TF SavedModel (production export)."""
    if len(sequence_data) < SEQ_LEN:
        return None

    # Shape: (1, 30, features)
    input_data = np.expand_dims(np.array(sequence_data, dtype=np.float32), axis=0)
    outputs = infer(**{input_key: tf.constant(input_data)})
    prediction = outputs[output_key].numpy()[0]  # (num_classes,)
    return prediction

try:
    dummy = np.zeros((1, SEQ_LEN, FEATURES), dtype=np.float32)
    _ = infer(**{input_key: tf.constant(dummy)})[output_key].numpy()
    logger.info("Model warmup OK.")
except Exception as e:
    logger.exception("Model warmup failed: %s", e)

# -------------------------
# Mediapipe Holistic (global + lock)
# -------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
    model_complexity=1,  # tweak: 0 faster, 2 more accurate
)
holistic_lock = threading.Lock()

# -------------------------
# Per-client state
# -------------------------
THRESHOLD = float(os.getenv("SIGN_THRESHOLD", "0.8"))
MAX_SENTENCE = int(os.getenv("MAX_SENTENCE", "5"))
PRED_SMOOTH_WINDOW = int(os.getenv("PRED_SMOOTH_WINDOW", "10"))

MAX_FPS = float(os.getenv("MAX_FPS", "15"))
MIN_FRAME_INTERVAL = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0

client_state = {}  # sid -> state dict
state_lock = threading.Lock()


def init_state():
    return {
        "sequence": deque(maxlen=SEQ_LEN),
        "sentence": deque(maxlen=MAX_SENTENCE),
        "predictions": deque(maxlen=PRED_SMOOTH_WINDOW),
        "last_ts": 0.0,
    }


# -------------------------
# Routes
# -------------------------
@app.route("/")
def hello_world():
    return {
        "status": "ok",
        "signature": signature_name,
        "input_key": input_key,
        "output_key": output_key,
        "actions": int(len(actions)),
        "threshold": THRESHOLD,
        "max_fps": MAX_FPS,
    }


# -------------------------
# Socket handlers
# -------------------------
@socketio.on("connect")
def handle_connect():
    sid = request.sid
    with state_lock:
        client_state[sid] = init_state()

    logger.info("Client connected sid=%s", sid)
    emit("connected", {"data": "Connected to server"})


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    with state_lock:
        client_state.pop(sid, None)
    logger.info("Client disconnected sid=%s", sid)


@socketio.on("video_frame")
def handle_video_frame(frame_data):
    sid = request.sid

    with state_lock:
        st = client_state.get(sid)
        if st is None:
            st = init_state()
            client_state[sid] = st

    now = time.time()
    if MIN_FRAME_INTERVAL > 0 and (now - st["last_ts"]) < MIN_FRAME_INTERVAL:
        return
    st["last_ts"] = now

    try:
        frame, err = decode_data_url_to_frame(frame_data)
        if err:
            emit("error", {"message": err})
            return
        
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with holistic_lock:
            results = holistic.process(frame_rgb)

        keypoints = extract_key_points(results)
        st["sequence"].append(keypoints)

        if len(st["sequence"]) < SEQ_LEN:
            emit("prediction", {
                "word": st["last_word"],
                "action": None,
                "confidence": 0.0,
                "status": f"collecting_frames_{len(st['sequence'])}/{SEQ_LEN}",
            })
            return

        res = make_prediction(st["sequence"])
        if res is None:
            return

        if len(res) != len(actions):
            msg = f"Model output classes ({len(res)}) != actions labels ({len(actions)}). Check labels.json/order."
            logger.error(msg)
            emit("error", {"message": msg})
            return

        pred_idx = int(np.argmax(res))
        predicted_action = str(actions[pred_idx])
        confidence = float(res[pred_idx])

        st["predictions"].append(pred_idx)

        if len(st["predictions"]) == st["predictions"].maxlen:
            if len(set(st["predictions"])) == 1 and confidence > THRESHOLD:
                if predicted_action != st["last_word"]:
                    st["last_word"] = predicted_action

        if predicted_action == "no_sign":
            emit("prediction", {
                "word": st["last_word"],
                "action": "no_sign",
                "confidence": confidence,
                "status": "no_sign",
            })
            return

        emit("prediction", {
            "word": st["last_word"],
            "action": predicted_action,
            "confidence": confidence,
            "status": "ok",
        })

    except Exception as e:
        logger.exception("Error processing frame sid=%s: %s", sid, e)
        emit("error", {"message": "Internal server error"})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)