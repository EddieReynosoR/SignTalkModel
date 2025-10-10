from flask import Flask
from flask_socketio import SocketIO, emit
from keras.api.models import load_model
import numpy as np
import mediapipe as mp
import cv2
import base64
import tensorflow as tf

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load and optimize model
def load_optimized_model():
    model = load_model("aprobar.h5")
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open('aprobar_optimized.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="aprobar_optimized.tflite")
    interpreter.allocate_tensors()
    
    return interpreter

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION
    image.flags.writeable = False # Image is no longer writeable
    results = model.process(image) # Make prediction
    image.flags.writeable = True # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION

    return image, results

def extract_key_points(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # We use 21 * 3 cause 21 is the total amount of data that we are going to receive
    # from a hand, and multiply by 3 cause of x y and z
    # It is the same case for the other landmarks

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])

def make_prediction(sequence_data):
    """Optimized prediction using TFLite"""
    if len(sequence_data) < 30:
        return None
    
    input_data = np.expand_dims(sequence_data[-30:], axis=0).astype(np.float32)
    
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])[0]
    
    return prediction

@app.route("/")
def hello_world():
    return "WebSocket Server Running"

model = load_optimized_model()
input_details = model.get_input_details()
output_details = model.get_output_details()

actions = np.array(["yo", "querer", "aprobar"])
sequence = []
sentence = []
predictions = []
threshold = 0.8

last_frame_time = 0
frame_interval = 0.1 

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on("video_frame")
def handle_video_frame(frame_data):
    """
    Handle incoming video frames from the frontend.
    """
    global sequence, sentence, predictions

    try:
         # Decode base64 string to image
        header, encoded = frame_data.split(",", 1)
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process frame using Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        # Extract keypoints
        keypoints = extract_key_points(results)

        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only last 30 frames
        
        # Make prediction when we have enough frames
        if len(sequence) == 30:
            res = make_prediction(list(sequence))
            predicted_action = actions[np.argmax(res)]
            predictions.append(np.argmax(res))
            
            # Visualization logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if predicted_action != sentence[-1]:
                            sentence.append(predicted_action)
                    else:
                        sentence.append(predicted_action)
            
            if len(sentence) > 5:
                sentence = sentence[-5:]
            
            # Send prediction results back to client
            emit('prediction', {
                'action': predicted_action,
                'confidence': float(res[np.argmax(res)]),
                'sentence': ' '.join(sentence)
            })
    except Exception as e:
        emit("error", {"message": str(e)})

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)