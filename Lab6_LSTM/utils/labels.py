"""Danh sach nhan dung cho cac bai phan loai."""

CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

FASHION_MNIST_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

BINARY_LABELS = {
    "catdog": ["cat", "dog"],
    "gender": ["female", "male"],
}

MODEL_PATHS = {
    "cifar10": "models/lstm_cifar10.h5",
    "catdog": "models/lstm_catdog.h5",
    "fashion_mnist": "models/lstm_fashion_mnist.h5",
    "gender": "models/lstm_gender.h5",
    "truyen_kieu": "models/lstm_truyen_kieu.h5",
    "twitter": "models/lstm_twitter.h5",
}

