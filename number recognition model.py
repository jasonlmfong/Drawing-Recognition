# import PIL
# import matplotlib.pyplot as plt
# import numpy as np
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def create_model():
    """build and train the AI model used later"""
    #directory
    train_dir = pathlib.Path("training")
    # print(train_dir)

    #total images
    # image_count = len(list(train_dir.glob("*/*.png")))
    # print(image_count)

    #show first image
    # ones = list(train_dir.glob("1/*"))
    # pic = PIL.Image.open(str(ones[0]))
    # pic.show()

    #separate training set and validation set
    batch_size = 32
    img_height = 400
    img_width = 525

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=102,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=102,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    #print all class names
    class_names = train_ds.class_names
    # print(class_names)

    #configure dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

    #standardize colors
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    #number of classes
    num_classes = 10

    #augmentation
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    #model creation with augmentation and dropout
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes)
    ])

    #model compilation
    model.compile(optimizer="adam",
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=["accuracy"])

    #training
    epochs=10
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs
    )

    return (model, img_height, img_width, class_names)

if __name__ == "__main__":
    (model, height, width, names) = create_model()

    # attempt to save the model
    model.save("trained model\saved_model")
    p = pathlib.Path("data_text")
    p.write_text(f"{height}*{width}*{names}")
