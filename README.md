# Handwriting Recognition System

This AI-powered system recognizes and classifies handwritten digits using **Convolutional Neural Networks (CNNs)**. Trained on the **MNIST dataset**, the model can identify handwritten digits and is easily extendable to recognize other characters or symbols.

## Key Features:

- **Image Preprocessing**: Converts input images to grayscale, applies thresholding, and resizes them to fit the model's input.
- **Model Training**: The CNN model is trained on the MNIST dataset to learn digit patterns.
- **Prediction**: The system predicts the digit from any new image and provides **softmax probabilities** for confidence levels.
- **Web Deployment**: The system is deployed via a **Flask** web application, where users can upload images in **base64 format** to receive predictions via a **RESTful API**.

## Technologies Used:

- **Python**
- **TensorFlow (Keras)**
- **Flask** (for web deployment)
- **MNIST dataset**
- **PIL (Pillow)** for image processing
- **OpenCV** (optional for advanced preprocessing)

## Use Cases:

- **Automatic number plate recognition** (for digits)
- **Digit recognition** in forms or checks
- **Assistive technology** for the visually impaired

This project demonstrates handwriting recognition using deep learning, and can be adapted for various applications requiring handwritten text recognition.
