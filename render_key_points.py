import cv2 # OpenCV dependency for computer vision
import numpy as np # NumPy for working with arrays of data, and how we struct different datasets
import os # provides a portable way of using OS functionalities (filepaths)
from matplotlib import pyplot as plt # For create visualizations in Python
import time # Implement sleep between frames
import mediapipe as mp # Mark lines in our body caption for sign language

import constants

# Keypoints using MP Holistic
mp_drawing = constants.mp_drawing

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION
    image.flags.writeable = False # Image is no longer writeable
    results = model.process(image) # Make prediction
    image.flags.writeable = True # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION

    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, constants.mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, constants.mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, constants.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, constants.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, constants.mp_holistic.FACEMESH_TESSELATION, 
    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, constants.mp_holistic.POSE_CONNECTIONS, 
    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, constants.mp_holistic.HAND_CONNECTIONS, 
    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, constants.mp_holistic.HAND_CONNECTIONS, 
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

    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # We use 21 * 3 cause 21 is the total amount of data that we are going to receive
    # from a hand, and multiply by 3 cause of x y and z
    # It is the same case for the other landmarks

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])

for action in constants.actions:
    if not os.path.exists(os.path.join(constants.DATA_PATH, action)):
        os.mkdir(os.path.join(constants.DATA_PATH, action))

for action in constants.actions: 
    # dirmax = np.max(np.array(os.listdir(os.path.join(constants.DATA_PATH, action))).astype(int))
    for sequence in range(constants.no_sequences):
        try: 
            # os.makedirs(os.path.join(constants.DATA_PATH, action, str(dirmax+sequence)))
            os.makedirs(os.path.join(constants.DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with constants.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in constants.actions:
        # Loop through sequences aka videos
        for sequence in range(constants.no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(constants.sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_key_points(results)
                npy_path = os.path.join(constants.DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()