const fileInput = document.getElementById('fileInput');
const predictButton = document.getElementById('predictButton');
const predictionResult = document.getElementById('predictionResult');

// Hàm chuyển đổi ảnh thành base64
function toBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);  // Loại bỏ phần đầu base64 prefix
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Hàm dự đoán
async function predict() {
    const file = fileInput.files[0];
    if (!file) {
        predictionResult.textContent = 'Vui lòng chọn hình ảnh.';
        return;
    }

    // Chuyển ảnh thành base64
    const base64Image = await toBase64(file);

    // Gửi yêu cầu đến server
    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {  // Đúng URL Flask server
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
        });

        const data = await response.json();

        if ('prediction' in data) {
            predictionResult.textContent = `Dự đoán: ${data.prediction}`;
        } else {
            predictionResult.textContent = 'Không thể nhận diện ảnh.';
        }
    } catch (error) {
        predictionResult.textContent = 'Đã xảy ra lỗi khi gửi ảnh!';
        console.error(error);
    }
}

predictButton.addEventListener('click', predict);
