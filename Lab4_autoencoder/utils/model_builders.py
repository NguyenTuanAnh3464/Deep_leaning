"""Cac kien truc Autoencoder va classifier dung lai cho 4 bai.

Code co y viet don gian de sinh vien de doc, de sua trong luc thuc hanh.
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
)


def build_dense_autoencoder(input_shape=(28, 28, 1), latent_dim=32, loss="mse"):
    """Tao Dense Autoencoder cho anh grayscale 28x28 nhu Fashion-MNIST."""
    inputs = Input(shape=input_shape, name="image_input")
    x = Flatten(name="flatten")(inputs)
    x = Dense(128, activation="relu", name="encoder_dense_128")(x)
    x = Dense(64, activation="relu", name="encoder_dense_64")(x)
    latent = Dense(latent_dim, activation="relu", name="latent_vector")(x)

    x = Dense(64, activation="relu", name="decoder_dense_64")(latent)
    x = Dense(128, activation="relu", name="decoder_dense_128")(x)
    x = Dense(input_shape[0] * input_shape[1] * input_shape[2], activation="sigmoid")(x)
    outputs = Reshape(input_shape, name="reconstructed_image")(x)

    autoencoder = Model(inputs, outputs, name="dense_autoencoder")
    encoder = Model(inputs, latent, name="dense_encoder")
    autoencoder.compile(optimizer="adam", loss=loss)
    return autoencoder, encoder


def build_conv_autoencoder(input_shape=(32, 32, 3), loss="mse"):
    """Tao Convolutional Autoencoder cho anh mau."""
    inputs = Input(shape=input_shape, name="image_input")

    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv_32")(inputs)
    x = MaxPooling2D((2, 2), padding="same", name="enc_pool_1")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv_64")(x)
    x = MaxPooling2D((2, 2), padding="same", name="enc_pool_2")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="enc_conv_128")(x)
    encoded = MaxPooling2D((2, 2), padding="same", name="latent_feature_map")(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="dec_conv_128")(encoded)
    x = UpSampling2D((2, 2), name="dec_up_1")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="dec_conv_64")(x)
    x = UpSampling2D((2, 2), name="dec_up_2")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv_32")(x)
    x = UpSampling2D((2, 2), name="dec_up_3")(x)
    outputs = Conv2D(input_shape[-1], (3, 3), activation="sigmoid", padding="same", name="reconstructed_image")(x)

    autoencoder = Model(inputs, outputs, name="conv_autoencoder")
    encoder = Model(inputs, encoded, name="conv_encoder")
    autoencoder.compile(optimizer="adam", loss=loss)
    return autoencoder, encoder


def build_classifier_from_encoder(
    encoder,
    input_shape,
    num_classes,
    binary=False,
    freeze_encoder=True,
):
    """Gan classifier phia sau encoder de nhan dang nhan anh."""
    encoder.trainable = not freeze_encoder

    inputs = Input(shape=input_shape, name="classifier_input")
    x = encoder(inputs)
    x = Flatten(name="classifier_flatten")(x)
    x = Dense(128, activation="relu", name="classifier_dense_128")(x)
    x = Dropout(0.3, name="classifier_dropout")(x)

    if binary:
        outputs = Dense(1, activation="sigmoid", name="classifier_output")(x)
        loss = "binary_crossentropy"
    else:
        outputs = Dense(num_classes, activation="softmax", name="classifier_output")(x)
        loss = "sparse_categorical_crossentropy"

    classifier = Model(inputs, outputs, name="autoencoder_classifier")
    classifier.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return classifier

