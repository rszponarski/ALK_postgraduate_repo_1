import os
import random
import numpy as np
import tensorflow as tf
import keras

# --- Stała wartość seed
seed = 42

# --- Ustawienie seedów dla reprodukowalności
os.environ["PYTHONHASHseed"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)

# --- Wymuszenie deterministycznych operacji w TensorFlow
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.config.experimental.enable_op_determinism()

# --- Parametryzacja
train_dir = "plants_train"
val_dir = "plants_test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 1. Wstępne przygotowanie danych

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model


train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 2. Tworzenie generatorów do danych

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
'''
# 3. Model bazowy

base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
)

base_model.trainable = False

# 4. Dodanie własnych klasyfikatorów

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="leaky_relu")(x)
outputs = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)'''

# 3. Model bazowy
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False  # Zamrożenie wag

# 4. Dodanie własnych klasyfikatorów
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  # relu zamiast leaky_relu
outputs = Dense(train_generator.num_classes, activation='softmax')(x)

'''Finalny model'''
model = Model(inputs=base_model.input, outputs=outputs)

# 5. Trening i zapisanie modelu

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_generator, validation_data=val_generator, epochs=5)
model.save("plant_seedings_model.h5")