import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BostonHousing:

    def prepare_data(self):
        """Load and prepare the Boston Housing dataset for training."""
        # Load dataset from OpenML (Boston Housing)
        boston = fetch_openml('boston', version=1)
        data = boston.data
        labels = boston.target

        # Split the data into training and testing sets
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Normalize the data (important for neural networks)
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        return train_data, train_labels, test_data, test_labels

    def build_model(self, input_shape):
        """Build and compile a simple neural network model for regression."""
        model = tf.keras.Sequential([    
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)  # Single output for regression task
        ])

        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

        return model

    def train(self, train_data, train_labels, model, epochs=20, batch_size=512, validation_split=0.2):
        """Train the model with training data and evaluate on a validation set."""
        history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        return history.history

    def plot_loss(self, history_dict):
        """Plot the training and validation loss."""
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
        """Evaluate the model on the test data."""
        results = model.evaluate(test_data, test_labels)
        return results

