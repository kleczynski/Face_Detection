import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os


learning_rate = 1 * 10**-4
epochs = 20
bs = 32

directory = r"C:\Users\klecz\Desktop\Computer-Vision-with-Python\Face_Detection\dataset"
categories = ['Face_Mask', 'No_Mask']

data = []
labels = []

for category in categories:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        try:
            img_arr = os.path.join(path, img)
            image = load_img(img_arr, target_size=(224,224))
            image = img_to_array(image)
            image = preprocess_input(image)

            data.append(image)
            labels.append(category)
        except Exception as e:
            pass

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="uint8")
labels = np.array(labels)


(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)


augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7,7))(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(128, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation='softmax')(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

for layer in base_model.layers:
    layer.trainable = False


opt = Adam(lr=learning_rate, decay=learning_rate / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

h = model.fit(
    x=augmentation.flow(x_train, y_train, batch_size=bs),
    steps_per_epoch=len(x_train) // bs,
    validation_data=(x_test, y_test),
    validation_steps=len(x_test),
    epochs=epochs
)

model.save("face_mask_detector.model", save_format="h5")


