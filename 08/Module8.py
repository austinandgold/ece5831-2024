import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from LeNet import LeNet
import sys

model_loader = LeNet()

# Custom keras file from LeNet
model_load = model_loader.load('Moore_cnn_model.keras')

# Image preprocessing functions
def increase_contrast(image):
    min_pixel = np.min(image)
    max_pixel = np.max(image)
    contrast_image = (image - min_pixel) * (255.0 / (max_pixel - min_pixel))
    contrast_image = np.clip(contrast_image, 0, 255)
    return contrast_image

def normalize_image(image):
    image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
    return image.astype(np.float32) / 255.0

def load_images_from_subfolders(image_folder, target_size=(28, 28)):
    image_arrays = []

    for subdir, _, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith('.png'):
                image_path = os.path.join(subdir, filename)
                try:
                    image = Image.open(image_path).convert('L')  
                    image = image.resize(target_size)
                    image = np.array(image)

                    image = 255.0 - image
                    image = increase_contrast(image)
                    image = normalize_image(image)

                    image_arrays.append(image.flatten())

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

    return np.array(image_arrays)

def display_image(image_numpy, index=0):
    if image_numpy.size == 0:
        print("No images to display.")
        return
    img_to_display = image_numpy[index].reshape(28, 28)
    plt.imshow(img_to_display, cmap='gray')
    plt.axis('off')
    plt.show()

def get_guess_for_filename(filename, predictions):
    index = int(filename.split('_')[0]) * 5 + int(filename.split('_')[1].split('.')[0])
    return predictions[index] if 0 <= index < len(predictions) else None

if __name__ == "__main__":
    image_folder = 'Custom MNIST Sample'
    images = load_images_from_subfolders(image_folder)

    batch_size = 16
    x_batch = images[:50]
    y_hat = model_loader.predict(x_batch)

    first_arg = sys.argv[1]
    second_arg = int(sys.argv[2])

    # Get the prediction for the specific image
    guess = get_guess_for_filename(first_arg, y_hat)

    if guess is not None:
        if second_arg == guess:
            print(f"Success: Image {first_arg} is correctly recognized as {guess}")
        else:
            print(f"Fail: Image {first_arg} was expected to be {second_arg}, but the prediction is {guess}")
    else:
        print(f"Failed to find a valid guess for image {first_arg}")