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

## Project Code Explanation

### 1. **app.py** (Flask Backend)
The `app.py` file is the backend of the Flask web application. It handles the image uploaded by the client, processes it, and sends the prediction result back to the client.

#### Key Functions:
- **Flask Setup**: 
    - The `Flask` module is used to create a web server.
    - `CORS` (Cross-Origin Resource Sharing) is enabled to allow the frontend to make requests from different domains.
    ```python
    app = Flask(__name__)
    CORS(app)
    ```

- **Model Loading**:
    - The machine learning model (a Convolutional Neural Network, or CNN) is loaded once at the start of the application to optimize performance and reduce the load time for each request.
    ```python
    model = tf.keras.models.load_model('minist_model.h5')
    print("✅ Model loaded successfully.")
    ```

- **Image Preprocessing**:
    - The image is received as base64 data, which is then decoded and converted into a `PIL` image.
    - **Grayscale Conversion**: The image is converted to grayscale to reduce complexity and focus on the relevant features (handwritten digits).
    - **Thresholding**: A thresholding technique is applied to convert the image to a clearer black-and-white format, making it easier for the model to recognize the characters. The threshold value can be adjusted to fine-tune this process:
    ```python
    image = image.point(lambda p: 200 if p < threshold_value else 0)
    ```
  
    **Thresholding Algorithm**:
    The thresholding function works by checking each pixel's value. If the pixel value is below the threshold, it is set to white (255); otherwise, it is set to black (0).
    \[
    f(x) = 
    \begin{cases}
    0 & \text{if pixel value} \, x > \text{threshold} \\
    255 & \text{if pixel value} \, x \leq \text{threshold}
    \end{cases}
    \]

- **Resizing the Image**:
    - The image is resized to 28x28 pixels, which is the standard input size for the MNIST dataset, where each image is 28x28 pixels in grayscale.
    ```python
    image_resized = image.resize((28, 28))
    ```

- **Prediction**:
    - The processed image is then converted into a NumPy array, normalized by dividing pixel values by 255 (scaling values to a range between 0 and 1).
    - The array is reshaped to match the input shape expected by the model (1, 28, 28, 1), where `1` represents the single channel (grayscale image).
    ```python
    img_array = np.array(image_resized) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    ```

    - The model then makes a prediction, and the digit with the highest probability is selected as the predicted result.
    ```python
    prediction = model.predict(img_array)
    predicted = int(np.argmax(prediction))
    ```

    **Softmax Function**:
    The output from the CNN model is a vector of probabilities, with each element representing the likelihood of the image belonging to a particular class (digit 0 to 9). The **softmax function** is used to convert raw model outputs (logits) into probabilities:
    \[
    P(y=i|x) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
    \]
    Where \(z_i\) is the raw output for class \(i\), and the sum in the denominator is over all classes.

### 2. **index.html** (Frontend)
The `index.html` file is the user interface of the web application. It allows users to upload an image and view the predicted result.

#### Key Features:
- **File Input**: The user can select an image file from their local device to be uploaded.
- **Button**: When the user clicks the "Predict" button, the image is sent to the Flask server for processing and prediction.
    ```html
    <input type="file" id="fileInput" accept="image/*" />
    <button id="predictButton">Predict</button>
    ```
- **Result Display**: The predicted digit is displayed for the user once the server responds with the prediction.
    ```html
    <p id="predictionResult">No prediction yet.</p>
    ```

### 3. **styles.css** (Styling)
The `styles.css` file defines the styling for the user interface. It ensures the application has a clean, user-friendly design.

#### Key Styles:
- **Centering Content**: The content is centered on the page using Flexbox, making the layout responsive and easy to navigate.
    ```css
    body {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
    }
    ```
- **Button Styling**: The "Predict" button is styled with a green color, rounded corners, and a hover effect to make it visually appealing.
    ```css
    button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
        border-radius: 5px;
    }
    button:hover {
        background-color: #45a049;
    }
    ```

### 4. **app.js** (Frontend Logic)
The `app.js` file handles the client-side logic of the web application, including image file handling, sending the image to the Flask server, and displaying the result.

#### Key Functions:
- **Image Conversion to Base64**: When the user selects an image, it is converted to base64 format for easy transmission to the server.
    ```javascript
    function toBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(',')[1]);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }
    ```

- **Prediction Request**: Once the image is converted to base64, it is sent to the Flask server via a `POST` request. The server processes the image and sends back the predicted result.
    ```javascript
    const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image })
    });
    ```

- **Display Prediction**: The predicted result is displayed on the webpage.
    ```javascript
    predictionResult.textContent = `Prediction: ${data.prediction}`;
    ```

### 5. **Main.ipynb** (Model Training)
The `Main.ipynb` file is used for training the machine learning model using the MNIST dataset, which contains handwritten digits. The model is then saved and used for prediction in the Flask app.

#### Key Steps:
- **MNIST Dataset**: The dataset is loaded using TensorFlow, and its pixel values are normalized.
    ```python
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    ```
- **CNN Architecture**: A Convolutional Neural Network (CNN) is built with multiple convolutional layers followed by max-pooling layers to extract features from the images.
    ```python
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    ```

- **Model Training**: The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function. It is then trained on the MNIST dataset.
    ```python
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ```

- **Saving the Model**: After training, the model is saved to a `.h5` file.
    ```python
    model.save('minist_model.h5')
    ```

This setup enables the Flask backend to use the trained model to recognize handwritten digits from images sent by the frontend.

