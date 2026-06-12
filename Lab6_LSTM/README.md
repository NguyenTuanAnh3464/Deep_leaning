# Bài thực hành Deep Learning: Long Short-Term Memory - LSTM

Do giáo viên không cung cấp dataset nên bài thực hành sử dụng dataset public tương đương. Project này cài đặt LSTM cho nhận dạng ảnh và sinh văn bản, đồng thời có Flask Web để thử dự đoán sau khi train model.

## 1. Giới thiệu bài thực hành

Project gồm 6 bài:

1. LSTM nhận dạng ảnh CIFAR10.
2. LSTM nhận dạng ảnh Cat/Dog.
3. LSTM nhận dạng ảnh Fashion-MNIST.
4. LSTM nhận dạng khuôn mặt Nam/Nữ.
5. LSTM sinh văn bản từ Truyện Kiều.
6. LSTM sinh văn bản Twitter.

Các bài được triển khai bằng Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Scikit-learn, Flask, Pillow và joblib.

## 2. LSTM là gì?

LSTM là một dạng mạng nơ-ron hồi quy dùng để học dữ liệu chuỗi. LSTM có các cổng Forget Gate, Input Gate và Output Gate giúp mô hình chọn thông tin cần quên, cần ghi nhớ và cần đưa ra ở từng bước thời gian.

Với ảnh, ta có thể xem mỗi hàng ảnh là một time step:

- Fashion-MNIST: `28` time steps, mỗi step `28` features.
- CIFAR10: `32` time steps, mỗi step `32 * 3 = 96` features.
- Cat/Dog và Nam/Nữ: resize `64x64x3`, có `64` time steps, mỗi step `64 * 3 = 192` features.

## 3. Cấu trúc project

```text
lstm_deeplearning_project/
├── LSTM_Thuc_Hanh_DeepLearning.ipynb
├── app.py
├── requirements.txt
├── README.md
├── datasets/
│   ├── truyen_kieu.txt
│   ├── twitter_data.csv
│   ├── catdog/
│   └── gender/
├── models/
├── tokenizers/
├── training/
│   ├── train_cifar10_lstm.py
│   ├── train_catdog_lstm.py
│   ├── train_fashion_mnist_lstm.py
│   ├── train_gender_lstm.py
│   ├── train_truyen_kieu_lstm.py
│   └── train_twitter_lstm.py
├── utils/
│   ├── image_preprocess.py
│   ├── text_preprocess.py
│   ├── labels.py
│   └── prediction.py
├── templates/
└── static/
```

## 4. Dataset sử dụng

- CIFAR10: TensorFlow/Keras, dùng `tensorflow.keras.datasets.cifar10.load_data()`.
- Fashion-MNIST: TensorFlow/Keras, dùng `tensorflow.keras.datasets.fashion_mnist.load_data()`.
- Cats vs Dogs: TensorFlow Datasets, dùng dataset `cats_vs_dogs`. Nếu không tải được, đặt ảnh thủ công vào `datasets/catdog/train/cat`, `datasets/catdog/train/dog`, `datasets/catdog/val/cat`, `datasets/catdog/val/dog`.
- Nam/Nữ: dùng FairFace hoặc dataset tự chuẩn bị, tách vào `datasets/gender/train/male`, `datasets/gender/train/female`, `datasets/gender/val/male`, `datasets/gender/val/female`.
- Truyện Kiều: đặt file text public vào `datasets/truyen_kieu.txt`. Project có file mẫu nhỏ để demo.
- Twitter: dùng Sentiment140 hoặc Twitter Sentiment Analysis Dataset, chuẩn hóa cột nội dung thành `twitter_content`. Project có CSV mẫu nhỏ để demo.

## 5. Cách cài thư viện

```bash
pip install -r requirements.txt
```

Nếu chạy trên Google Colab, nên bật GPU:

```text
Runtime > Change runtime type > GPU
```

## 6. Cách chạy notebook

Upload `LSTM_Thuc_Hanh_DeepLearning.ipynb` lên Google Colab hoặc mở bằng Jupyter Notebook/JupyterLab. Chạy từng cell theo thứ tự từ trên xuống.

## 7. Cách train từng mô hình

Chạy từ thư mục `lstm_deeplearning_project`:

```bash
python training/train_cifar10_lstm.py
python training/train_catdog_lstm.py
python training/train_fashion_mnist_lstm.py
python training/train_gender_lstm.py
python training/train_truyen_kieu_lstm.py
python training/train_twitter_lstm.py
```

Mỗi file có biến `EPOCHS = 3` để train nhanh. Có thể tăng lên `20`, `50` hoặc `100` nếu cần kết quả tốt hơn.

Sau khi train, model được lưu vào:

- `models/lstm_cifar10.h5`
- `models/lstm_catdog.h5`
- `models/lstm_fashion_mnist.h5`
- `models/lstm_gender.h5`
- `models/lstm_truyen_kieu.h5`
- `models/lstm_twitter.h5`

Tokenizer cho bài sinh văn bản được lưu vào:

- `tokenizers/tokenizer_truyen_kieu.pkl`
- `tokenizers/tokenizer_twitter.pkl`

## 8. Cách chạy Flask Web

Chạy từ thư mục `lstm_deeplearning_project`:

```bash
python app.py
```

Sau đó mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

Nếu chưa có model, web sẽ hiện thông báo: `Chưa có model. Vui lòng chạy file train tương ứng trước.`

## 9. Nhận xét kết quả

LSTM phù hợp nhất với dữ liệu chuỗi. Khi áp dụng cho ảnh, ta biến mỗi hàng ảnh thành một time step để mô hình học theo chuỗi hàng ảnh. Fashion-MNIST thường dễ hơn CIFAR10 vì ảnh grayscale và đối tượng đơn giản hơn. CIFAR10 khó hơn vì ảnh màu, nhiều lớp và nền ảnh phức tạp.

Cat/Dog và Nam/Nữ phụ thuộc nhiều vào chất lượng dataset, số lượng ảnh và cách chia train/validation. Với sinh văn bản, model cần nhiều dữ liệu và nhiều epochs hơn để câu tự nhiên. Twitter có nhiều ký tự nhiễu, URL, mention và hashtag nên cần làm sạch dữ liệu trước khi train.

