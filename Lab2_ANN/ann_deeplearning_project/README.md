# Bài thực hành Deep Learning: Artificial Neural Network

Project này cài đặt 5 mô hình ANN bằng Python, TensorFlow/Keras và triển khai demo dự đoán bằng Flask.

## Mục tiêu

- Hiểu cách xây dựng mô hình Artificial Neural Network bằng `Sequential`.
- Thực hành tiền xử lý dữ liệu ảnh và dữ liệu bảng.
- Train, đánh giá, vẽ biểu đồ accuracy/loss và lưu model.
- Triển khai ứng dụng web để upload ảnh hoặc nhập form dự đoán.

## Cấu trúc project

```text
ann_deeplearning_project/
├── ANN_Thuc_Hanh_DeepLearning.ipynb
├── app.py
├── requirements.txt
├── README.md
├── models/
├── training/
├── utils/
├── templates/
├── static/
└── datasets/
```

Thư mục `models/` ban đầu chưa có file `.h5`. Sau khi train, các script sẽ tự lưu model và encoder/preprocessor vào đây.

## Dataset sử dụng

- CIFAR10: có sẵn trong TensorFlow/Keras.
- MNIST: có sẵn trong TensorFlow/Keras.
- Cats vs Dogs: TensorFlow Datasets, tên dataset `cats_vs_dogs`.
- Adult Income: UCI Machine Learning Repository, ưu tiên package `ucimlrepo`.
- Car Evaluation: UCI Machine Learning Repository, ưu tiên package `ucimlrepo`.

## Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Nếu dùng Google Colab, có thể chạy:

```python
!pip install tensorflow-datasets ucimlrepo joblib
```

## Train từng model

Chạy các lệnh từ thư mục gốc `ann_deeplearning_project/`:

```bash
python training/train_cifar10_ann.py
python training/train_mnist_ann.py
python training/train_catdog_ann.py
python training/train_adult_ann.py
python training/train_car_ann.py
```

Model được lưu vào:

- `models/ann_cifar10.h5`
- `models/ann_mnist.h5`
- `models/ann_catdog.h5`
- `models/ann_adult.h5`
- `models/ann_car.h5`

Biểu đồ accuracy/loss được lưu trong `static/training_plots/`.

## Chạy Flask

```bash
python app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

Nếu web báo `Chưa có model. Vui lòng chạy file train tương ứng trước.`, hãy chạy file train của bài đó trước.

## Cat/Dog local dataset

Nếu `tensorflow_datasets` không tải được `cats_vs_dogs`, hãy chuẩn bị dữ liệu theo cấu trúc:

```text
datasets/catdog/
├── train/
│   ├── cat/
│   └── dog/
└── val/
    ├── cat/
    └── dog/
```

## Chạy trên Google Colab

1. Upload file `ANN_Thuc_Hanh_DeepLearning.ipynb`.
2. Chọn `Runtime > Change runtime type > GPU` nếu có.
3. Chạy các cell theo thứ tự từ trên xuống.
4. Nếu cần chạy Flask trên Colab, nên dùng thêm `pyngrok` hoặc chạy các phần train/model trong notebook trước.

## Cơ sở lý thuyết ANN

ANN gồm nhiều tầng neuron. Trong bài này, đầu vào được đưa qua các tầng `Dense`, mỗi tầng học một biểu diễn mới của dữ liệu. Hidden layer dùng activation `relu` để học quan hệ phi tuyến, output layer dùng:

- `softmax` cho phân loại nhiều lớp: CIFAR10, MNIST, Car Evaluation.
- `sigmoid` cho phân loại nhị phân: Cat/Dog, Adult Income.

`Dropout` được thêm vào để giảm overfitting. Với dữ liệu ảnh, vì đây là bài ANN nên ảnh được flatten thành vector 1 chiều trước khi đưa vào mạng, không dùng `Conv2D` hoặc `MaxPooling2D`.

## Tiền xử lý dữ liệu

- Ảnh CIFAR10, MNIST, Cat/Dog: chuẩn hóa pixel về khoảng `[0, 1]`, sau đó flatten.
- Adult Income: xử lý missing value, one-hot encoding cột categorical, chuẩn hóa cột numeric bằng `StandardScaler`.
- Car Evaluation: one-hot encoding toàn bộ input categorical, encode nhãn output bằng `LabelEncoder`.

## Kiến trúc mô hình

| Bài | Dataset | Input shape | Kiến trúc chính | Loss | File model |
| --- | --- | --- | --- | --- | --- |
| CIFAR10 | TensorFlow/Keras | 3072 | Dense 512, Dropout, Dense 256, Dropout, Dense 10 | sparse_categorical_crossentropy | `models/ann_cifar10.h5` |
| MNIST | TensorFlow/Keras | 784 | Dense 256, Dropout, Dense 128, Dense 10 | sparse_categorical_crossentropy | `models/ann_mnist.h5` |
| Cat/Dog | TensorFlow Datasets | 12288 | Dense 256, Dropout, Dense 128, Dropout, Dense 1 | binary_crossentropy | `models/ann_catdog.h5` |
| Adult | UCI | sau one-hot | Dense 128, Dropout, Dense 64, Dense 1 | binary_crossentropy | `models/ann_adult.h5` |
| Car | UCI | sau one-hot | Dense 64, Dropout, Dense 32, Dense 4 | sparse_categorical_crossentropy | `models/ann_car.h5` |

## Nhận xét và kết luận

ANN có thể xử lý cả dữ liệu ảnh và dữ liệu bảng nếu tiền xử lý đúng định dạng đầu vào. Tuy nhiên với ảnh, ANN flatten làm mất thông tin không gian nên thường kém CNN. Trong phạm vi bài thực hành, mục tiêu chính là hiểu quy trình train model: load dataset, tiền xử lý, xây dựng mạng Dense, train, đánh giá, lưu model và triển khai dự đoán bằng Flask.
