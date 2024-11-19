import numpy as np
import os
import cv2

from sklearn.model_selection import train_test_split # Allow us to partition our data into a training partition and test it on a different segment of data
from keras.src.utils import to_categorical

from keras.src.models import Sequential # Sequential neural network
from keras.src.layers import LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
# LSTM layer for use a temporal component to work with neural network, and allow us to perform action detection
from keras.src.callbacks import TensorBoard # Allow us to monitoring our neural network model
from keras.src.callbacks import ModelCheckpoint
from keras.src.regularizers import L2

import mediapipe as mp

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Path for exported data, numpy arrays
DATA_PATH = os.path.join("MP_Data")

# Actions that we try to detect
actions = np.array(["yo", "querer", "aprobar"])
# 30 videos of data
no_sequences = 40
# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION
    image.flags.writeable = False # Image is no longer writeable
    results = model.process(image) # Make prediction
    image.flags.writeable = True # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION

    return image, results

def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
    # mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    # mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    # )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

# The input data used in the detection model is a series of 30 arrays
# each of which contains 1662 values (30, 1662)
# Each of the 30 arrays represents the landmark values (1662 values)
# from a single array.

# pose = []
# for res in results.pose_landmarks.landmark:
#     test = np.array([res.x, res.y, res.z, res.visibility])
#     pose.append(test)

def extract_key_points(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # We use 21 * 3 cause 21 is the total amount of data that we are going to receive
    # from a hand, and multiply by 3 cause of x y and z
    # It is the same case for the other landmarks

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    # return np.concatenate([pose, face, lh, rh])
    return np.concatenate([pose, lh, rh])

#region PROCESS DATA AND CREATE LABELS AND FEATURES
# Create a label dictionary for each one of our labels
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#endregion

#region BUILD AND TRAIN LSTM Neural Network

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30,258), kernel_regularizer=L2(0.01), recurrent_regularizer=L2(0.01)))
model.add(Dropout(0.2))

model.add(LSTM(128, return_sequences=True, activation='tanh', kernel_regularizer=L2(0.01), recurrent_regularizer=L2(0.01)))
model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=False, activation='tanh', kernel_regularizer=L2(0.01), recurrent_regularizer=L2(0.01)))
model.add(Dropout(0.2))

model.add(Dense(64, activation='tanh', kernel_regularizer=L2(0.01)))
model.add(Dropout(0.2))

model.add(Dense(32, activation='tanh', kernel_regularizer=L2(0.01)))
model.add(Dense(actions.shape[0], activation='softmax'))

cp_best_val_loss = ModelCheckpoint(
      "SLD_val_loss.weights.h5", monitor='val_loss', mode = 'min', save_weights_only=True, save_best_only=True, verbose=1
)
cp_best_val_acc = ModelCheckpoint(
      "SLD_val_acc.weights.h5", monitor='val_categorical_accuracy', mode = 'max', save_weights_only=True, save_best_only=True, verbose=1
)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=2000, validation_data = (x_test, y_test), callbacks = [cp_best_val_loss, cp_best_val_acc])

model.save("aprobar.h5")

model.load_weights("SLD_val_loss.weights.h5")

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
#endregion


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):

    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.999

# Use OpenCV for using the device camera
cap = cv2.VideoCapture(0)

# Set mediapipe model, with keyword is like using in C#, is for cleaning up automatically that resource
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_key_points(results)
        # sequence.insert(0,keypoints)
        # sequence = sequence[:30]
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res):
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]          
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
#endregion