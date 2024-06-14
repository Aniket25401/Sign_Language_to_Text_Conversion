from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import cv2
import numpy as np
import time

# Load model architecture and weights file
json_file = open("modelword.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("modelword.h5")

# List of colors used in visualization
colors = []
for i in range(0, 20):
    colors.append((245, 117, 16))

# Draw rectangle box which represents the probability of gesture and overlay labels
def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Hand keypoints, recognized word, predicted accuracy, threshold, and detected word are stored here
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8
detected_word = ""
previous_word = ""
last_letter_time = time.time()
same_letter_counter = 0  # Counter for the same detected letter
consecutive_frames_threshold = 30  # Threshold for consecutive frames

cap = cv2.VideoCapture(0)
# mediapipe library to setup hand tracking.
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    # Opens webcam and starts to capture video frames from camera in loop
    while cap.isOpened():
        ret, frame = cap.read()
        # Crops the frame to focus on region to work on
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)

        # Call the modified mediapipe_detection() function
        image_landmarked, results = mediapipe_detection(cropframe, hands)
        # Keypoints are extracted and last 30 frames worth
        # Keypoints are stored in sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Checks if 30 frames are present in sequence
        try:
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                # checks if the predicted class is stable for last 10 frames
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        last_letter_time = time.time()  # Reset the timer
                        detected_letter = actions[np.argmax(res)]
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)] * 100))
                                detected_word += detected_letter  # Append detected letter to the word
                                same_letter_counter = 1  # Reset counter
                            else:
                                same_letter_counter += 1
                                if same_letter_counter >= consecutive_frames_threshold:
                                    same_letter_counter = 0
                                    detected_word += detected_letter  # Append detected letter to the word
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))
                            detected_word += detected_letter  # Append detected letter to the word
                            same_letter_counter = 1  # Reset counter
                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]
        except Exception as e:
            pass

        # Check if no letter detected for 10 seconds and reset the word
        if time.time() - last_letter_time > 10:
            if detected_word:
                if previous_word:
                    previous_word += " " + detected_word
                else:
                    previous_word = detected_word
                detected_word = ""
                sentence = []
                accuracy = []

        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: -" + ' '.join(sentence) + ''.join(accuracy), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Word: " + detected_word, (10, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Sentence: " + previous_word, (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Overlay the image with hand landmarks drawn on the cropframe
        frame[40:400, 0:300] = image_landmarked
        cv2.imshow('OpenCV Feed', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
