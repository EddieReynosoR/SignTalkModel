import cv2
import json
import time
from dataclasses import dataclass
from typing import Any, Tuple, List
import numpy as np
from pathlib import Path
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

@dataclass
class CaptureConfig:
    data_path: Path
    word: str
    no_sequences: int = 30
    sequence_length: int = 30
    camera_index: int = 0
    include_face: bool = False
    min_det_conf: float = 0.5
    min_trk_conf: float = 0.5
    warmup_seconds: float = 1.0
    frame_skip: int = 0
    draw_landmarks: bool = True
    countdown_each_sequence: int = 3

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def slugify(text: str) -> str:
    return "".join(c for c in text.strip().lower().replace(" ", "_") if c.isalnum() or c in ("_", "-"))

def mediapipe_detection(image: np.ndarray, model) -> Tuple[np.ndarray, Any]:
    """Convierte a RGB, procesa y regresa BGR + resultados."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results

def draw_landmarks(image: np.ndarray, results: Any) -> None:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.face_landmarks is not None:
        mp_drawing.draw_landmarks(image, results.face_landmarks)

def extract_key_points(results: Any, include_face: bool) -> np.ndarray:
    # Pose: 33 puntos * (x,y,z,vis) = 132
    pose = (np.array([[lm.x, lm.y, lm.z, lm.visibility]
                      for lm in getattr(results.pose_landmarks, "landmark", [])],
                     dtype=np.float32).flatten()
            if results.pose_landmarks else np.zeros(33 * 4, dtype=np.float32))

    # Face opcional: 468 * (x,y,z) = 1404
    if include_face:
        face = (np.array([[lm.x, lm.y, lm.z]
                          for lm in getattr(results.face_landmarks, "landmark", [])],
                         dtype=np.float32).flatten()
                if results.face_landmarks else np.zeros(468 * 3, dtype=np.float32))
    else:
        face = np.empty(0, dtype=np.float32)

    # Hands: 21 * (x,y,z) = 63 cada una
    lh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in getattr(results.left_hand_landmarks, "landmark", [])],
                   dtype=np.float32).flatten()
          if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32))

    rh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in getattr(results.right_hand_landmarks, "landmark", [])],
                   dtype=np.float32).flatten()
          if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32))

    return np.concatenate([pose, face, lh, rh])

def write_metadata(seq_dir: Path, cfg: CaptureConfig, sequence_idx: int) -> None:
    meta = {
        "word": cfg.word,
        "sequence_index": sequence_idx,
        "sequence_length": cfg.sequence_length,
        "include_face": cfg.include_face,
        "timestamp": int(time.time()),
        "version": "v1.0"
    }
    (seq_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def linspace_indices(n_src: int, n_dst: int) -> np.ndarray:
    if n_src <= 1:
        return np.zeros(n_dst, dtype=int)
    xs = np.linspace(0, n_src - 1, num=n_dst)
    return np.clip(np.round(xs).astype(int), 0, n_src - 1)

def resample_sequence(seq: list[np.ndarray], target_len: int) -> np.ndarray:
    idx = linspace_indices(len(seq), target_len)
    return np.stack([seq[i] for i in idx], axis=0)

def ema_smooth(seq: List[np.ndarray], alpha: float = 0.35) -> List[np.ndarray]:
    if not seq: 
        return seq
    out = [seq[0].copy()]
    for t in range(1, len(seq)):
        out.append(alpha * seq[t] + (1 - alpha) * out[-1])
    return out

def capture_for_seconds(cap, holistic, seconds: float, draw: bool, include_face: bool) -> list[np.ndarray]:
    start = time.time()
    frames = []
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        image, results = mediapipe_detection(frame, holistic)
        if draw:
            draw_landmarks(image, results)
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
        kp = extract_key_points(results, include_face)
        frames.append(kp)
    return frames

def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "MP_Data"
    data_path.mkdir(parents=True, exist_ok=True)

    word = input("Indica la palabra que vas a escanear: ").strip()
    if not word:
        raise ValueError("Debes indicar una palabra válida.")

    cfg = CaptureConfig(
        data_path=data_path,
        word=slugify(word),
        no_sequences=30,
        sequence_length=30,
        include_face=False,
        camera_index=0,
        min_det_conf=0.5,
        min_trk_conf=0.5,
        warmup_seconds=1.0,
        frame_skip=0,
        draw_landmarks=True,
        countdown_each_sequence=2
    )

    word_dir = cfg.data_path / cfg.word
    ensure_dir(word_dir)

    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        print(f"Warm-up {cfg.warmup_seconds}s…")
        time.sleep(cfg.warmup_seconds)

        with mp_holistic.Holistic(
            min_detection_confidence=cfg.min_det_conf,
            min_tracking_confidence=cfg.min_trk_conf
        ) as holistic:

            for sequence in range(cfg.no_sequences):
                seq_dir = word_dir / str(sequence)
                ensure_dir(seq_dir)

                # Cuenta regresiva visual por secuencia
                start_countdown = time.time()
                while time.time() - start_countdown < cfg.countdown_each_sequence:
                    ret, frame = cap.read()
                    if not ret:
                        print("Frame no leído durante la cuenta atrás; reintentando…")
                        continue
                    cv2.putText(frame, 'STARTING COLLECTION', (60, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, f'Palabra: {cfg.word}  Secuencia: {sequence+1}/{cfg.no_sequences}',
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                        return

                write_metadata(seq_dir, cfg, sequence)

                saved = 0
                frame_idx = 0
                while saved < cfg.sequence_length:
                    ret, frame = cap.read()
                    if not ret:
                        print("Frame no leído; continuando…")
                        continue

                    image, results = mediapipe_detection(frame, holistic)

                    if cfg.draw_landmarks:
                        draw_landmarks(image, results)

                    cv2.putText(image, f'Capturando {cfg.word} | Seq {sequence+1}/{cfg.no_sequences} '
                                       f'| Frame {saved+1}/{cfg.sequence_length}',
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                    # Saltar frames si se indicó
                    if cfg.frame_skip > 0 and (frame_idx % (cfg.frame_skip + 1)) != 0:
                        frame_idx += 1
                        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                            return
                        continue

                    keypoints = extract_key_points(results, cfg.include_face)
                    # Guardar SIEMPRE con extensión .npy
                    npy_path = seq_dir / f"{saved}.npy"
                    np.save(npy_path, keypoints)

                    saved += 1
                    frame_idx += 1

                    # Salida suave
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        return

        print("Captura finalizada correctamente.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
