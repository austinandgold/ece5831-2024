from mnist_data import MnistData
import numpy as np
import gzip
import urllib.request
import pickle
import os
import matplotlib.pyplot as plt
import sys

#using if statement to separate the test vs train date
if sys.argv[1] == "test":

    #index of label
    second_argument = sys.argv[2]
    mnist_data = MnistData()
    (_, _), (test_images, test_labels) = mnist_data.load()

    #convert argv to int and show image and label (one-hot-encoded)
    plt.imshow(mnist_data.dataset['test_images'][int(second_argument)].reshape(28,28))
    print(f" Label: {mnist_data.dataset['test_labels'][int(second_argument)]}")
    plt.show()

if sys.argv[1] == "train":

    second_argument = 1
    mnist_data = MnistData()
    (_, _), (train_images, train_labels) = mnist_data.load()

    plt.imshow(mnist_data.dataset['train_images'][int(second_argument)].reshape(28,28))
    print(f" Label: {mnist_data.dataset['train_labels'][int(second_argument)]}")
    plt.show()