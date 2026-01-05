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


def extract_key_points(results) -> np.ndarray:
    pose = (np.array([[lm.x, lm.y, lm.z, lm.visibility]
                      for lm in getattr(results.pose_landmarks, "landmark", [])],
                     dtype=np.float32).flatten()
            if results.pose_landmarks else np.zeros(33 * 4, dtype=np.float32))

    face = np.empty(0, dtype=np.float32)

    lh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in getattr(results.left_hand_landmarks, "landmark", [])],
                   dtype=np.float32).flatten()
          if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32))

    rh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in getattr(results.right_hand_landmarks, "landmark", [])],
                   dtype=np.float32).flatten()
          if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32))

    return np.concatenate([pose, face, lh, rh]).astype(np.float32)


def decode_data_url_to_frame(frame_data: str):
    if not isinstance(frame_data, str) or not frame_data:
        return None, "Empty frame payload"

    try:
        if "," in frame_data:
            _, b64 = frame_data.split(",", 1)
        else:
            b64 = frame_data

        b64 = b64.strip()
        image_data = base64.b64decode(b64, validate=False)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, "Could not decode frame"
        return frame, None
    except Exception as e:
        return None, f"Decode error: {e}"


base_dir = Path(__file__).resolve().parent
DATA_PATH = base_dir / "MP_Data"

actions = np.array(
    sorted([f for f in os.listdir(DATA_PATH) if (DATA_PATH / f).is_dir()]),
    dtype=str,
)

saved_model = tf.saved_model.load(str(base_dir / "sign_model"))
signature_name = "serving_default" if "serving_default" in saved_model.signatures else next(iter(saved_model.signatures))
infer = saved_model.signatures[signature_name]

input_key = next(iter(infer.structured_input_signature[1].keys()))
output_key = next(iter(infer.structured_outputs.keys()))

logger.info("SavedModel signature=%s, input_key=%s, output_key=%s", signature_name, input_key, output_key)

FEATURES = 33 * 4 + 21 * 3 + 21 * 3
SEQ_LEN = 30


def make_prediction(sequence_data: deque):
    if len(sequence_data) < SEQ_LEN:
        return None

    input_data = np.expand_dims(np.array(sequence_data, dtype=np.float32), axis=0)
    outputs = infer(**{input_key: tf.constant(input_data)})
    prediction = outputs[output_key].numpy()[0]
    return prediction


try:
    dummy = np.zeros((1, SEQ_LEN, FEATURES), dtype=np.float32)
    _ = infer(**{input_key: tf.constant(dummy)})[output_key].numpy()
    logger.info("Model warmup OK.")
except Exception as e:
    logger.exception("Model warmup failed: %s", e)


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
    model_complexity=1,
)
holistic_lock = threading.Lock()


THRESHOLD = float(os.getenv("SIGN_THRESHOLD", "0.8"))
MAX_SENTENCE = int(os.getenv("MAX_SENTENCE", "5"))
PRED_SMOOTH_WINDOW = int(os.getenv("PRED_SMOOTH_WINDOW", "10"))

TOP_RATIO = float(os.getenv("TOP_RATIO", "0.8"))
COOLDOWN_SEC = float(os.getenv("COOLDOWN_SEC", "0.9"))

NO_SIGN_STREAK_FRAMES = int(os.getenv("NO_SIGN_STREAK", "10"))
MARGIN_RELAX = float(os.getenv("MARGIN_RELAX", "0.25"))
RELAX_THRESHOLD = float(os.getenv("RELAX_THRESHOLD", "0.65"))

MIN_MARGIN_ACCEPT = float(os.getenv("MIN_MARGIN_ACCEPT", "0.12"))
CLEAR_ON_ACCEPT = os.getenv("CLEAR_ON_ACCEPT", "1") == "1"

MAX_FPS = float(os.getenv("MAX_FPS", "30"))
MIN_FRAME_INTERVAL = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0

MAX_FRAME_WIDTH = int(os.getenv("MAX_FRAME_WIDTH", "1280"))

client_state = {}
state_lock = threading.Lock()


def init_state():
    return {
        "lock": threading.Lock(),
        "sequence": deque(maxlen=SEQ_LEN),
        "sentence": deque(maxlen=MAX_SENTENCE),
        "predictions": deque(maxlen=PRED_SMOOTH_WINDOW),
        "last_ts": 0.0,
        "last_word": "no_sign",
        "last_sentence_word": "",
        "cooldown_until": 0.0,
        "no_sign_streak": 0,
        "ready_for_word": True,
    }


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
        "min_margin_accept": MIN_MARGIN_ACCEPT,
        "clear_on_accept": CLEAR_ON_ACCEPT,
        "no_sign_streak_frames": NO_SIGN_STREAK_FRAMES,
    }


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

    try:
        now = time.time()
        with st["lock"]:
            if MIN_FRAME_INTERVAL > 0 and (now - st["last_ts"]) < MIN_FRAME_INTERVAL:
                return
            st["last_ts"] = now

        frame, err = decode_data_url_to_frame(frame_data)
        if err:
            emit("error", {"message": err})
            return

        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH:
            scale = float(MAX_FRAME_WIDTH) / float(w)
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        with holistic_lock:
            results = holistic.process(frame_rgb)

        keypoints = extract_key_points(results)

        with st["lock"]:
            st["sequence"].append(keypoints)
            last_word = st.get("last_word", "no_sign")
            seq_len = len(st["sequence"])

        if seq_len < SEQ_LEN:
            emit(
                "prediction",
                {
                    "word": last_word,
                    "action": None,
                    "confidence": 0.0,
                    "status": f"collecting_frames_{seq_len}/{SEQ_LEN}",
                    "sentence": list(st.get("sentence", [])),
                },
            )
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

        top2 = float(np.partition(res, -2)[-2]) if len(res) >= 2 else 0.0
        margin = float(res[pred_idx] - top2)
        dyn_thr = RELAX_THRESHOLD if margin > MARGIN_RELAX else THRESHOLD

        accepted_word = None

        with st["lock"]:
            now2 = time.time()

            if predicted_action == "no_sign":
                st["no_sign_streak"] = st.get("no_sign_streak", 0) + 1
                if st["no_sign_streak"] >= NO_SIGN_STREAK_FRAMES:
                    st["ready_for_word"] = True
                    st["predictions"].clear()
            else:
                st["no_sign_streak"] = 0
                if now2 >= st.get("cooldown_until", 0.0):
                    st["predictions"].append((pred_idx, confidence))

            if len(st["predictions"]) == st["predictions"].maxlen:
                idxs = [p[0] for p in st["predictions"]]
                mode_idx = max(set(idxs), key=idxs.count)
                ratio = idxs.count(mode_idx) / len(idxs)
                confs = [c for (i, c) in st["predictions"] if i == mode_idx]
                avg_conf = (sum(confs) / len(confs)) if confs else 0.0
                mode_action = str(actions[mode_idx])

                if (
                    st.get("ready_for_word", True)
                    and now2 >= st.get("cooldown_until", 0.0)
                    and ratio >= TOP_RATIO
                    and avg_conf >= dyn_thr
                    and margin >= MIN_MARGIN_ACCEPT
                ):
                    if mode_action != "no_sign" and mode_action != st.get("last_word", "no_sign"):
                        st["last_word"] = mode_action
                        st["cooldown_until"] = now2 + COOLDOWN_SEC
                        st["ready_for_word"] = False

                        if mode_action != st.get("last_sentence_word", ""):
                            st["sentence"].append(mode_action)
                            st["last_sentence_word"] = mode_action

                        accepted_word = mode_action

                        if CLEAR_ON_ACCEPT:
                            st["sequence"].clear()
                            st["predictions"].clear()

            word_out = st.get("last_word", "no_sign")
            sentence_out = list(st.get("sentence", []))

        emit(
            "prediction",
            {
                "word": word_out,
                "action": predicted_action,
                "confidence": confidence,
                "status": "no_sign" if predicted_action == "no_sign" else "ok",
                "sentence": sentence_out,
                "accepted": accepted_word,
                "margin": margin,
                "threshold": dyn_thr,
            },
        )

    except Exception as e:
        logger.exception("Error processing frame sid=%s: %s", sid, e)
        emit("error", {"message": "Internal server error"})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)