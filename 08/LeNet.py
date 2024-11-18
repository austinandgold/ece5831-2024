from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
from PIL import Image

class LeNet:
    def __init__(self, batch_size=32, epochs=20):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self._create_lenet()
        self._compile()
    

    def _create_lenet(self):
        self.model = Sequential([
            Conv2D(filters=6, kernel_size=(5,5), 
                   activation='sigmoid', input_shape=(28, 28, 1), 
                   padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            
            Conv2D(filters=16, kernel_size=(5,5), 
                   activation='sigmoid', 
                   padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),

            Flatten(),

            Dense(120, activation='sigmoid'),
            Dense(84, activation='sigmoid'),
            Dense(10, activation='softmax')
        ])

    def _compile(self):
        if self.model is None:
            print('Error: Create a model first..')
        
        self.model.compile(optimizer='Adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        

    def _preprocess(self):
        # load mnist data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # normalize
        x_train = x_train/255.0
        x_test = x_test/255.0

        # add channel dim
        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  

        # one-hot encoding
        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

    def train(self):
        self._preprocess()
        self.model.fit(self.x_train, self.y_train, 
                  batch_size=self.batch_size, 
                  epochs=self.epochs)

    def save(self, model_path_name):
     
        try:
            # Ensure the directory exists
            directory = os.path.dirname(model_path_name)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Save the model as a .keras file
            self.model.save(model_path_name)
            print(f"Model saved to {model_path_name}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load(self, model_path_name):
    
        try:
            # Load the model from the .keras file
            self.model = tensorflow.keras.models.load_model(model_path_name)
            print(f"Model loaded from {model_path_name}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict(self, images):
            """
            Predict the class labels for the input images.

            Args:
                images (list or np.array): A list or array of image data.

            Returns:
                list: Predicted class probabilities or class labels.
            """
            try:
                # If images is a list, convert it to a numpy array
                if isinstance(images, list):
                    images = np.array(images)
                
                # Ensure the images are in the correct format (e.g., resized, normalized)
                # If images are raw (e.g., not preprocessed), preprocess them
                
                # Preprocessing images (e.g., resizing to match input shape of model)
                processed_images = []
                for img in images:
                    # If the image is a PIL image or a numpy array, resize it to the expected input size
                    # Assuming the model was trained on images of size (32, 32, 3)
                    if isinstance(img, np.ndarray):
                        img = image.array_to_img(img)  # Convert numpy array to a PIL image if needed
                    
                    # Resize the image to match model's input size (e.g., 32x32)
                    img = img.resize((32, 32))  # Change the size if your model needs a different size
                    img = image.img_to_array(img)  # Convert PIL image back to a numpy array
                    
                    # Normalize the image (if needed, e.g., values between 0 and 1)
                    img = img / 255.0  # Assuming the model was trained with normalized images

                    # Append to the list of processed images
                    processed_images.append(img)

                # Convert list of images into a numpy array for prediction
                processed_images = np.array(processed_images)

                # Make predictions with the model
                predictions = self.model.predict(processed_images)
                
                # If model outputs class probabilities, you can get the predicted class labels:
                predicted_labels = np.argmax(predictions, axis=-1)
                
                return predicted_labels  # or return predictions if you need probabilities

            except Exception as e:
                print(f"Error making predictions: {e}")
                return None
