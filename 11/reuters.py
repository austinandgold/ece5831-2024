import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    
    #Vectorize sequences
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.0
    return results

class Reuters:

    def prepare_data(self):
        
        # Load dataset
        (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

        # Vectorize both train and test data
        x_train = vectorize_sequences(train_data)
        x_test = vectorize_sequences(test_data)

        # Convert labels to one-hot encoded format
        y_train = to_categorical(train_labels)
        y_test = to_categorical(test_labels)

        return x_train, y_train, x_test, y_test

    def build_model(self):
        #Build and compile model
        model = tf.keras.Sequential([    
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10000,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(46, activation='softmax')  # 46 classes for Reuters dataset
        ])

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, x_train, y_train, model, epochs=20, batch_size=512, validation_split=0.2):
        
        # Train the model with training data and evaluate on a validation set
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

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

    def plot_accuracy(self, history_dict):
        #Plot the training and validation accuracy
        plt.clf()
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def evaluate(self, model, x_test, y_test):
        #Evaluate the model
        results = model.evaluate(x_test, y_test)
        return results
