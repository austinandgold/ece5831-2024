import numpy as np
import gzip
import urllib.request
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import mnist
import sys
import matplotlib as mpl

mnist = mnist.Mnist()
(_, _), (test_images, test_labels) = mnist.load()
net = mnist.init_network()

first_arg = sys.argv[1]
second_arg = sys.argv[2]

img = Image.open(f'Custom MNIST Sample/Digit {second_arg}/{first_arg}').convert('L')
img = img.resize((28,28))
img = np.array(img)
img = 255.0 - img
img = (img - np.min(img))*(255/(np.max(img)-np.min(img)))
img = img.astype(np.float32)/255
img = img.flatten()

x = img

y_hat = mnist.predict(x)
p = np.argmax(y_hat)

a = img.reshape(28,28)

if __name__ == "__main__":
    '''This enables the user to enter a file path in the command line for analysis, also show image of sample'''
    sample = Image.open(f'Custom MNIST Sample/Digit {second_arg}/{first_arg}')
    sample.show()
    if int(second_arg) == p:
        print(f'Success: Image {first_arg} is for digit {second_arg} is recognized by {p}')
    else:
        print(f'Fail: Image {first_arg} is for digit {second_arg} but the inference result is {p}') 