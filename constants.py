import os
import numpy as np
import mediapipe as mp

# Use mediapipe for marking the areas that we want to use for sign language
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Path for exported data, numpy arrays
DATA_PATH = os.path.join("MP_Data")

# Actions that we try to detect
actions = np.array(["yo", "querer", "aprobar"])
# 30 videos of data
no_sequences = 30
# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30