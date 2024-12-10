import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#ended up using the housing data from scikit-learn module, was having an issue accessing the data from keras.datasets

class BostonHousing:

    def prepare_data(self):

        # Load dataset
        boston = fetch_openml('boston', version=1)
        data = boston.data
        labels = boston.target

        # Split the data
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        return train_data, train_labels, test_data, test_labels

    def build_model(self, input_shape):

        # Single output for regression task
        model = tf.keras.Sequential([    
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

        return model

    def train(self, train_data, train_labels, model, epochs=20, batch_size=512, validation_split=0.2):

        #training the model
        history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        return history.history

    def plot_loss(self, history_dict):
        #Plot the training and validation loss
        plt.clf()
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, 'b-.', label='Training Loss')
        plt.plot(epochs, val_loss_values, 'b-', label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def evaluate(self, model, test_data, test_labels):
        #Evaluate the model
        results = model.evaluate(test_data, test_labels)
        return results

