from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Tải mô hình một lần
model = tf.keras.models.load_model('minist_model.h5')
print("✅ Mô hình đã được tải thành công.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📥 Nhận dữ liệu từ client")

        if 'image' not in data:
            return jsonify({'error': 'Thiếu ảnh'}), 400

        image_data = base64.b64decode(data['image'])

        try:
            image = Image.open(BytesIO(image_data))
        except UnidentifiedImageError:
            return jsonify({'error': 'Ảnh không hợp lệ'}), 400

        # Xử lý ảnh
        image = image.convert('L')  # Grayscale

        # Tạo ngưỡng để chuyển ảnh thành đen và trắng rõ ràng hơn (thresholding)
        threshold_value = 200  # Giá trị ngưỡng tùy chỉnh (có thể thay đổi)
        image = image.point(lambda p: 200 if p < threshold_value else 0)

        # Resize ảnh về kích thước 28x28
        image_resized = image.resize((28, 28))

        # Chuyển ảnh thành mảng numpy và chuẩn hóa giá trị pixel về [0, 1]
        img_array = np.array(image_resized) / 255.0

        # Reshape ảnh cho phù hợp với đầu vào của mô hình (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)

        print(f"📊 Pixel range: {img_array.min():.4f} - {img_array.max():.4f}")

        # Dự đoán với mô hình
        prediction = model.predict(img_array)
        predicted = int(np.argmax(prediction))
        print(f"🔢 Softmax: {np.round(prediction[0], 3)}")
        print(f"✅ Dự đoán: {predicted}")

        return jsonify({'prediction': predicted})

    except Exception as e:
        print(f"❌ Lỗi server: {str(e)}")
        return jsonify({'error': 'Lỗi server.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
