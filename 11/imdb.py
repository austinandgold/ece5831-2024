import tensorflow as tf
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt

class Imdb:

    def prepare_data(self):
        # Load dataset
        (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=10000)

        # Vectorize sequences (turn them into binary matrices)
        def vectorize_sequences(sequences, dimension=10000):
            results = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                for j in sequence:
                    results[i, j] = 1.0
            return results
        
        # Vectorize both train and test data
        x_train = vectorize_sequences(x_train)
        x_test = vectorize_sequences(x_test)

        # Convert labels to float32 type
        y_train = np.asarray(y_train).astype('float32')
        y_test = np.asarray(y_test).astype('float32')

        # Return the processed data
        return x_train, y_train, x_test, y_test

    def build_model(self):
        # Build the model
        model = tf.keras.Sequential([    
            tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Return the model
        return model

    def train(self, x_train, y_train, model):
        # Split data for validation
        x_val = x_train[:10000]
        y_val = y_train[:10000]

        x_train_partial = x_train[10000:]
        y_train_partial = y_train[10000:]

        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
            tf.keras.callbacks.TensorBoard()
        ]

        # Train the model
        history = model.fit(x_train_partial, y_train_partial, epochs=20, batch_size=512,
                            validation_data=(x_val, y_val), callbacks=callbacks)

        # Return the history dictionary for plotting
        return history.history

    def plot_loss(self, history_dict):
        # Plot loss values
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
        # Plot accuracy values
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
        # Evaluate the model on the test data
        results = model.evaluate(x_test, y_test)

        return results
