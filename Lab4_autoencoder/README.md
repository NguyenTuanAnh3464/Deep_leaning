# Bài thực hành Deep Learning: Autoencoder

Project này cài đặt Autoencoder để học đặc trưng ảnh, tái tạo ảnh và kết hợp Encoder với Dense classifier để nhận dạng nhãn. Code được viết đơn giản để sinh viên có thể chạy trên máy cá nhân hoặc Google Colab.

## 1. Giới thiệu bài thực hành

Bài thực hành gồm 4 bài toán:

- CIFAR10: nhận dạng 10 lớp ảnh màu.
- Cat/Dog: nhận dạng ảnh mèo hoặc chó.
- Fashion-MNIST: nhận dạng 10 lớp thời trang.
- Nam/Nữ: nhận dạng ảnh khuôn mặt nam hoặc nữ từ dataset tự chuẩn bị.

Ngoài các file train, project có Flask web app để upload ảnh và dự đoán bằng model đã train.

## 2. Autoencoder là gì?

Autoencoder là mô hình học không giám sát gồm 2 phần chính:

- Encoder: nén ảnh đầu vào thành biểu diễn đặc trưng có kích thước nhỏ hơn.
- Decoder: tái tạo lại ảnh từ biểu diễn đặc trưng đó.

Trong bài này, Autoencoder không dùng trực tiếp để phân loại. Sau khi train Autoencoder, ta lấy Encoder làm bộ trích xuất đặc trưng và gắn thêm Dense classifier để nhận dạng nhãn ảnh.

## 3. Cấu trúc project

```text
autoencoder_deeplearning_project/
├── AutoEncoder_Thuc_Hanh_DeepLearning.ipynb
├── app.py
├── requirements.txt
├── README.md
├── models/
├── datasets/
├── training/
│   ├── train_cifar10_autoencoder.py
│   ├── train_catdog_autoencoder.py
│   ├── train_fashion_autoencoder.py
│   └── train_gender_autoencoder.py
├── utils/
│   ├── labels.py
│   ├── model_builders.py
│   └── preprocess.py
├── templates/
│   ├── index.html
│   └── result.html
└── static/
    ├── uploads/
    ├── training_plots/
    └── style.css
```

Các file `.h5` trong `models/` sẽ được tạo sau khi chạy các file train.

## 4. Dataset sử dụng

- CIFAR10: TensorFlow/Keras, dùng `tensorflow.keras.datasets.cifar10.load_data()`.
- Fashion-MNIST: TensorFlow/Keras, dùng `tensorflow.keras.datasets.fashion_mnist.load_data()`.
- Cat/Dog: ưu tiên TensorFlow Datasets `cats_vs_dogs`.
- Nam/Nữ: dùng FairFace hoặc dataset tự chuẩn bị theo thư mục `male/female`.

Cấu trúc dataset local cho Cat/Dog:

```text
datasets/catdog/train/cat
datasets/catdog/train/dog
datasets/catdog/val/cat
datasets/catdog/val/dog
```

Cấu trúc dataset local cho Nam/Nữ:

```text
datasets/gender/train/male
datasets/gender/train/female
datasets/gender/val/male
datasets/gender/val/female
```

## 5. Cách cài thư viện

```bash
pip install -r requirements.txt
```

Nếu chạy trên Google Colab, nên bật GPU:

```text
Runtime > Change runtime type > GPU
```

## 6. Cách train từng mô hình

Train CIFAR10:

```bash
python training/train_cifar10_autoencoder.py
```

Train Cat/Dog:

```bash
python training/train_catdog_autoencoder.py
```

Train Fashion-MNIST:

```bash
python training/train_fashion_autoencoder.py
```

Train Nam/Nữ:

```bash
python training/train_gender_autoencoder.py
```

Mặc định mỗi file train dùng `EPOCHS = 5` để test nhanh. Nếu muốn so sánh 50, 100, 200 epochs, mở file train tương ứng và đổi:

```python
RUN_EPOCH_COMPARISON = True
```

Các model được lưu:

```text
models/ae_cifar10_autoencoder.h5
models/ae_cifar10_encoder.h5
models/ae_cifar10_classifier.h5
models/ae_catdog_autoencoder.h5
models/ae_catdog_encoder.h5
models/ae_catdog_classifier.h5
models/ae_fashion_autoencoder.h5
models/ae_fashion_encoder.h5
models/ae_fashion_classifier.h5
models/ae_gender_autoencoder.h5
models/ae_gender_encoder.h5
models/ae_gender_classifier.h5
```

Biểu đồ loss và ảnh tái tạo được lưu trong:

```text
static/training_plots/
```

## 7. Cách chạy Flask Web

Sau khi train ít nhất một model, chạy:

```bash
python app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

Nếu model chưa tồn tại, web sẽ hiển thị:

```text
Chưa có model. Vui lòng chạy file train tương ứng trước.
```

## 8. Cách chạy trên Google Colab

1. Upload `AutoEncoder_Thuc_Hanh_DeepLearning.ipynb` lên Google Colab.
2. Chọn `Runtime > Change runtime type > GPU` nếu có.
3. Chạy từng cell theo thứ tự.
4. Với Cat/Dog, Colab có thể cần thời gian tải `tensorflow_datasets`.
5. Với Nam/Nữ, upload dataset vào đúng cấu trúc thư mục `datasets/gender/...`.

## 9. Nhận xét kết quả

- Autoencoder học cách nén và tái tạo ảnh.
- Epochs tăng thường giúp ảnh tái tạo rõ hơn nhưng thời gian train lâu hơn.
- CIFAR10 khó hơn Fashion-MNIST vì ảnh màu, nhiều đối tượng và nền phức tạp.
- Cat/Dog và Nam/Nữ phụ thuộc nhiều vào chất lượng, số lượng và độ cân bằng dataset.
- Autoencoder không phải mô hình phân loại trực tiếp, cần kết hợp Encoder với classifier để nhận dạng nhãn.

