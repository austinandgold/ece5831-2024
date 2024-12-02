import pathlib
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from keras import layers
import datetime

#Parameters
def __init__(self):
    self.train_dataset = train_dataset
    self.validation_dataset = validation_dataset
    self.test_dataset = test_dataset
    self.model = None


def make_dataset_folders(self, subset_name, start_index, end_index):
    for category in ("dog", "cat"):
        dir = base_dir / subset_name / category
        #print(dir)
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        files = [f'{category}.{i}.jpg' for i in range(start_index, end_index)]
        #print(files)
        for i, file in enumerate(files):
            shutil.copyfile(src=src_dir / file, dst=dir / file)
            if i % 100 == 0: # show only once every 100
                print(f'src:{src_dir / file} => dst:{dir / file}')

def _make_dataset(self, subset_name):
    
    "subset_name"_dataset = tf.keras.utils.image_dataset_from_directory(
    base_dir / subset_name,
    image_size=(180,180),
    batch_size=32
)
    
def build_network(self, augmentation=True)
    
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ])

    inputs = layers.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=2)(x)
    #x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    #x = layers.AveragePooling2D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(200, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

def train(self, model_name):
    
    model_name.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.keras'),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    history = model_name.fit(train_dataset, epochs=20, 
                    validation_data=validation_dataset, 
                    callbacks=callbacks)