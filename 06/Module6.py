import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
import sys
import pickle

my_weight_pkl_file = 'Moore_mnist_model.pkl'

# Initialize network
network = TwoLayerNetWithBackProp(input_size=28*28, hidden_size=100, output_size=10)

with open(f'{my_weight_pkl_file}', 'rb') as f:
    network.params = pickle.load(f)


network.update_layers()
def increase_contrast(image):
    """Increase contrast using contrast stretching."""
    min_pixel = np.min(image)
    max_pixel = np.max(image)
    contrast_image = (image - min_pixel) * (255.0 / (max_pixel - min_pixel))
    contrast_image = np.clip(contrast_image, 0, 255)
    return contrast_image

def normalize_image(image):
    """Normalize image to the range [0, 1]."""
    image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
    return image.astype(np.float32) / 255.0

def load_images_from_subfolders(image_folder, target_size=(28, 28)):
    """Load images from subfolders and process them."""
    image_arrays = []

    for subdir, _, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith('.png'):
                image_path = os.path.join(subdir, filename)

                try:
                    image = Image.open(image_path).convert('L')  # Convert to grayscale
                    image = image.resize(target_size)
                    image = np.array(image)

                    # Preprocess the image
                    image = 255.0 - image  # Invert the image
                    image = increase_contrast(image)
                    image = normalize_image(image)

                    # Flatten the image and append to the list
                    image_arrays.append(image.flatten())

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

    return np.array(image_arrays)

def display_image(image_numpy, index=0):
    """Displays an image using matplotlib."""
    if image_numpy.size == 0:
        print("No images to display.")
        return
    img_to_display = image_numpy[index].reshape(28, 28)
    plt.imshow(img_to_display, cmap='gray')
    plt.axis('off')
    plt.show()

def get_guess_for_filename(filename, predictions):
    """Map filename to its corresponding prediction."""
    index = int(filename.split('_')[0]) * 5 + int(filename.split('_')[1].split('.')[0])
    return predictions[index] if 0 <= index < len(predictions) else None

# Example usage
if __name__ == "__main__":
    image_folder = 'Custom MNIST Sample'  # Path to image folder
    images = load_images_from_subfolders(image_folder)

    # Make predictions for the first 50 images (you can adjust this batch size)
    batch_size = 16
    x_batch = images[:50]
    y_hat = network.predict(x_batch)

    # Print the entire matrix of guesses (indices of the predicted classes)
    guesses = np.argmax(y_hat, axis=1)  # Get the predicted class labels (indices)
    print("Predicted class labels for the batch of images:")
    print(guesses)  # Print the matrix of guesses (flattened)

    first_arg = sys.argv[1]
    second_arg = sys.argv[2]

    # Get the prediction for the image filename
    guess = get_guess_for_filename(first_arg, y_hat)

    # Ensure `guess` is a scalar value (the predicted class label)
    if guess is not None:
        guess = np.argmax(guess)  # Get the index of the predicted class
        if int(second_arg) == guess:
            print(f"Success: Image {first_arg} is correctly recognized as {guess}")
        else:
            print(f"Fail: Image {first_arg} was expected to be {second_arg}, but the prediction is {guess}")
    else:
        print(f"Failed to find a valid guess for image {first_arg}")

