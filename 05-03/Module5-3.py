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
mnist.init_network()

#first_arg = sys.argv[1]
#second_arg = int(sys.argv[2])

img = Image.open("Custom MNIST Sample/Digit 2/2_2.png").convert('L')
img = img.resize((28,28))
img = np.array(img)
img = 255.0 - img
img = (img - np.min(img))*(255/(np.max(img)-np.min(img)))
img = img.astype(np.float32)/255
img = img.flatten()

x = img

y_hat = mnist.predict(x)

a = img.reshape(28,28)

if __name__ == "__main__":
    '''This enables the user to enter a file path in the command line for analysis, also show image of sample'''
    sample = Image.open('Custom MNIST Sample/Digit 2/2_2.png')
    sample.show()