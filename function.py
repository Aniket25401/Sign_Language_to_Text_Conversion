#import dependency
import cv2
import numpy as np
import os
import mediapipe as mp

#Utility module for drawing landmarks and connections.
mp_drawing = mp.solutions.drawing_utils
#Provides default styles for drawing landmarks.
mp_drawing_styles = mp.solutions.drawing_styles
#Provides hand tracking model and configurations.
mp_hands = mp.solutions.hands

#This function processes an image to detect hands and draw landmarks using the Mediapipe model.
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    image_landmarked = image.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image_landmarked, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return image_landmarked, results

#Draws hand landmarks with styled connections on the image.
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

#Extracts keypoints from hand landmarks and flattens them into a single array
def extract_keypoints(results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
        return(np.concatenate([rh]))
      
# Path for exported data, numpy arrays
#Defines where to save the extracted data and how many sequences and frames to collect for each gesture.
DATA_PATH = os.path.join('MP_Data') 

actions = np.array(['E','H','L','O','R','U','W'])

no_sequences = 30

sequence_length = 30