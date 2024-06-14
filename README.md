# Sign_Language_to_Text_Conversion

This project aims to develop a system that can recognize hand gestures from sign language and convert them into text in real time. It utilizes deep learning techniques, specifically LSTM neural networks, to accurately identify and interpret sign language gestures.

## Overview
The system is designed to capture hand gestures using a webcam, process the video frames to extract hand landmarks, and then use a trained neural network model to classify these gestures into corresponding text outputs.

## Installation

### Prerequisites
- Python 3.6+ (3.9.0 preferred)
- Virtual Environment (optional but recommended)

### Setup
1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/sign-language-to-text.git
    cd sign-language-to-text
    ```

2. **Create a Virtual Environment** (optional but recommended):
    - On Windows:
      ```sh
      python -m venv venv
      .\venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```sh
      python3 -m venv venv
      source venv/bin/activate
      ```

3. **Install Dependencies**:
    ```
    pip install -r requirements.txt
    ```

## Usage

### Data Collection
**Run the Data Collection Script**:
    **```
    collect_data.py
    ```**
    The dataset consists of images of hand gestures, which are collected using the collect_data.py script. This script captures images from the webcam and saves them in designated directories for each gesture.

### Model Training
**Train the Model**:
    **```
    train.py
    ```**
    This script will train the model using the collected images and save the trained model. This includes the following steps:

  - Loading and preprocessing the dataset.
  - Building the LSTM-based neural network.
  - Training the model on the preprocessed data.
  - Saving the trained model for later use.

### Real-Time Prediction
**Run the Real-Time Prediction Script**:
    **```
    app.py
    ```**
    This script is used for real-time gesture recognition and text conversion. It captures video from the webcam, processes the frames to       extract hand landmarks, and uses the trained model to predict the gesture being shown. The predicted gestures are then displayed as text on       the screen.

## Project Structure
- **`collect_data.py`** Script for collecting hand gesture images
- **`functionn.py`** Utility functions for processing images and keypoints
- **`train.py`** Script for training the model
- **`app.py`** Script for real-time gesture recognition
- **`requirements.txt`** List of project dependencies
- **`modelword.h5`** Trained model file
- **`modelword.json`** Model architecture file
- **`README.md`** Project README file

## Evaluation
To evaluate the model's performance, the **`train.py`** script prints the training and testing accuracy. Further evaluation metrics such as confusion matrices and classification reports can be added to analyze the model in more detail.

## Improvements
Several improvements can be made to enhance the system's accuracy and robustness:

- **Data Augmentation**: Apply transformations like rotations and shifts to increase the diversity of training data.
- **Hyperparameter Tuning**: Experiment with different hyperparameters for better model performance.
- **Regularization**: Use techniques like dropout and early stopping to prevent overfitting.
- **Transfer Learning**: Leverage pre-trained models to enhance feature extraction.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the **LICENSE** file for details.

