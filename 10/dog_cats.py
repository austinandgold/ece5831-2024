import os
import shutil
import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image


class DogsCats:
    
    def __init__(self, model=None, train_dataset=None, validation_dataset=None, test_dataset=None, base_dir=None, src_dir=None):
        """
        Initialize the DogsCats class. If a model is provided, datasets are optional.
        If no model is provided, datasets must be passed.

        Args:
        - model (tf.keras.Model, optional): Pre-trained model (if provided, datasets are optional).
        - train_dataset (tf.data.Dataset, optional): Training dataset (required if no model).
        - validation_dataset (tf.data.Dataset, optional): Validation dataset (required if no model).
        - test_dataset (tf.data.Dataset, optional): Testing dataset (required if no model).
        - base_dir (Path, optional): Path to the base directory for datasets.
        - src_dir (Path, optional): Path to the source directory for dataset files.
        """

        # Set model directly if provided
        self.model = model

        # If no model is provided, datasets must be provided
        if not self.model:
            if train_dataset is None or validation_dataset is None or test_dataset is None:
                raise ValueError("You must provide train, validation, and test datasets if no model is provided.")
            self.train_dataset = train_dataset
            self.validation_dataset = validation_dataset
            self.test_dataset = test_dataset
        else:
            # If the model is provided, datasets are not required
            self.train_dataset = None
            self.validation_dataset = None
            self.test_dataset = None

        # Set base_dir and src_dir if provided, else use defaults
        self.base_dir = base_dir or Path("path/to/base_dir")  # Default path if not provided
        self.src_dir = src_dir or Path("path/to/src_dir")  # Default path if not provided

        # Print out the directory information for debugging purposes (optional)
        print(f"Base directory is set to: {self.base_dir}")
        print(f"Source directory is set to: {self.src_dir}")
        
    def make_dataset_folders(self, subset_name, start_index, end_index):

        for category in ("dog", "cat"):
            dir_path = self.base_dir / subset_name / category
            # Create the category directory if it doesn't exist
            dir_path.mkdir(parents=True, exist_ok=True)
            files = [f'{category}.{i}.jpg' for i in range(start_index, end_index)]
            
            # Copy files from source to target directory
            for i, file in enumerate(files):
                shutil.copyfile(src=self.src_dir / file, dst=dir_path / file)
                if i % 100 == 0:  # Print every 100th file for progress
                    print(f'Copied: {self.src_dir / file} => {dir_path / file}')

    def make_dataset(self, subset_name):

        return tf.keras.utils.image_dataset_from_directory(
            self.base_dir / subset_name,
            image_size=(180, 180),
            batch_size=32
        )

    def build_network(self, augmentation=True):

        if augmentation:
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip('horizontal'),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2)
            ])
        else:
            data_augmentation = None

        inputs = layers.Input(shape=(180, 180, 3))
        x = inputs
        if data_augmentation:
            x = data_augmentation(x)
        x = layers.Rescaling(1./255)(x)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.AveragePooling2D(pool_size=2)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(200, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, model, epochs=20):

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.keras'),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]

        history = model.fit(self.train_dataset, epochs=epochs, 
                            validation_data=self.validation_dataset, 
                            callbacks=callbacks)

        return history  # Return the history object for further analysis

    def load_model(self, model_path_name):
  
        try:
            # Load the model from the .keras file
            self.model = tf.keras.models.load_model(model_path_name)
            print(f"Model loaded from {model_path_name}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict(self, image_file):

        try:
            # Load the image and resize it to the model's expected input size (180x180)
            img = image.load_img(image_file, target_size=(180, 180))
            
            # Convert the image to a numpy array
            img_array = image.img_to_array(img)
            
            # Normalize the image (scale pixel values to [0, 1])
            img_array = img_array / 255.0  # Assuming the model was trained with normalized images
            
            # Add an extra dimension to the image array (to represent batch size)
            img_array = np.expand_dims(img_array, axis=0)

            # Use the model to predict the class (0 or 1)
            predictions = self.model.predict(img_array)
            
            # If the model outputs class probabilities, convert them to binary class labels
            predicted_class = (predictions > 0.5).astype(int)  # Threshold 0.5 for binary classification
            
            # Show the image using Matplotlib
            plt.imshow(img)
            plt.axis('off')  # Hide axes
            plt.show()

            # Print the prediction result
            if predicted_class == 0:
                print(f"The image is predicted as: Cat")
            else:
                print(f"The image is predicted as: Dog")

        except Exception as e:
            print(f"Error making prediction: {e}")