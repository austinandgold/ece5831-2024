import tensorflow as tf
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt

class imdb:

    def prepare_data(self):

        (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=10000)

        def vectorize_sequences(sequences, dimension=10000):
            results = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                for j in sequence:
                    results[i, j] = 1.0
            
            return results
        
        x_train = vectorize_sequences(x_train)
        x_test = vectorize_sequences(x_test)

        y_train = np.asarray(y_train).astype('float32')
        y_test = np.asarray(y_test).astype('float32')



    def build_model(self):

        model = tf.keras.Sequential([    
            
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        

    def train(self, x_train, y_train, model):

        x_val = x_train[:10000]
        y_val = y_train[:10000]

        x_train_partial = x_train[10000:]
        y_train_partial = y_train[10000:]

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2
            ),
            tf.keras.callbacks.TensorBoard()
        ]

        history = model.fit(x_train_partial, y_train_partial, epochs=20, batch_size=512,
                    validation_data=(x_val, y_val), callbacks=callbacks)

        history_dict = history.history

    def plot_loss(self, history_dict):
        
        plt.clf()
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epoches = range(1, len(loss_values) + 1)

        plt.plot(epoches, loss_values, 'b-.', label='Training Loss')
        plt.plot(epoches, val_loss_values, 'b-', label='Validation Loss')
        plt.legend()
        plt.show()

    def plot_accuracy(self, history_dict):
        
        plt.clf()
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        epoches = range(1, len(acc) + 1)

        plt.plot(epoches, acc, 'bo', label='Training acc')
        plt.plot(epoches, val_acc, 'b', label='Validation acc')
        plt.legend()
        plt.show()
        
    def evaluate(self,model,x_test,y_test):

        results = model.evaluate(x_test, y_test)

        return results