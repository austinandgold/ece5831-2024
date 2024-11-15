import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
import sys


network = TwoLayerNetWithBackProp(input_size=28*28, hidden_size=100, output_size=10)

def increase_contrast(image):
    """Increase contrast using contrast stretching."""
    # Find the minimum and maximum pixel values in the image
    min_pixel = np.min(image)
    max_pixel = np.max(image)
    
    # Apply contrast stretching (stretching pixel values to full [0, 255] range)
    contrast_image = (image - min_pixel) * (255.0 / (max_pixel - min_pixel))
    
    # Clip values to ensure they remain within the valid range [0, 255]
    contrast_image = np.clip(contrast_image, 0, 255)
    
    return contrast_image

def load_images_from_subfolders(image_folder, target_size=(28, 28)):
    # List to hold all image arrays
    image_arrays = []

    # Loop through the folders and files recursively
    for subdir, _, files in os.walk(image_folder):
        for filename in files:
            # Only process files that end with '.PNG' (case-sensitive)
            if filename.endswith('.png'):
                print(f"Processing file: {filename}")  # Debugging line
                # Create the full path to the image file
                image_path = os.path.join(subdir, filename)

                try:
                    # Open the image using PIL and convert it to grayscale ('L' mode)
                    image = Image.open(image_path).convert('L')  # Convert to grayscale

                    # Resize the image to the target size (e.g., 28x28)
                    image = image.resize(target_size)

                    # Convert the resized image to a numpy array
                    image = np.array(image)

                    # Invert the image colors (255 - pixel value)
                    image = 255.0 - image

                    # Increase the contrast of the image
                    image = increase_contrast(image)

                    # Normalize the image to range [0, 255]
                    image = (image - np.min(image)) * (255 / (np.max(image) - np.min(image)))

                    # Convert to float32 and scale the values to [0, 1]
                    image = image.astype(np.float32) / 255.0

                    # Flatten the image
                    image = image.flatten()

                    # Append the processed image to the list
                    image_arrays.append(image)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")  # Print any error

    # Convert the list of image arrays into a single NumPy array (shape: num_images, flattened_pixels)
    if len(image_arrays) == 0:
        print("No images were processed.")
    else:
        print(f"Processed {len(image_arrays)} images.")

    images_numpy = np.array(image_arrays)

    return images_numpy

def display_image(image_numpy, index=0):
    """Displays an image using matplotlib."""
    # Check if the image_numpy array has any images
    if image_numpy.size == 0:
        print("No images to display.")
        return

    # Select an image by index from the numpy array
    img_to_display = image_numpy[index].reshape(28, 28)  # Grayscale images have shape (28, 28)

    # Display the image using matplotlib
    plt.imshow(img_to_display, cmap='gray')  # Use 'gray' colormap for grayscale images
    plt.axis('off')  # Hide the axes for better presentation
    plt.show()

# Example usage:
image_folder = 'Custom MNIST Sample'  # Replace with the path to your image folder
images = load_images_from_subfolders(image_folder)

# Display the first image from the array (you can change the index to display a different image)
#display_image(images, index=0)

# Access the image_numpy array directly
# print("Shape of image_numpy:", images.shape)  # Prints the shape of the image_numpy array

# Access the first image (flattened) directly:
# first_image = images[0]
# print("First image (flattened):", first_image)

# Access the first image in original shape (28x28)
# first_image_reshaped = first_image.reshape(28, 28)
# print("First image reshaped (28x28):\n", first_image_reshaped)

if __name__ == "__main__":
    y_hat = network.predict(images[0:50])

    batch_size = 16
    x = images[0:50]
    #y = test_labels[0:10]

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_hat_batch = network.predict(x_batch)
        p = np.argmax(y_hat_batch, axis=1)
        print(p)

    #first_arg = sys.argv[1]
    #second_arg = sys.argv[2]