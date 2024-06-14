from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import os
import numpy as np

# Creates a dictionary label_map to map gesture labels to numerical indices. 
# It uses the enumerate function to assign a unique numerical index to each label in the actions list.
label_map = {label: num for num, label in enumerate(actions)}

# To store sequences and labels
sequences, labels = [], []

# Each action, sequence and frame to load data is stored as numpy array
# Data and their label is then appended to their lists
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Converts the list of sequences and labels into numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Splits data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# TensorBoard logs during training
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Initializes sequential neural network model using keras
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compiling the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=140, callbacks=[tb_callback])

# Evaluating the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Saving the model
model_json = model.to_json()
with open("modelWord.json", "w") as json_file:
    json_file.write(model_json)
model.save('modelWord.h5')
