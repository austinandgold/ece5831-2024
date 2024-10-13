import cv2
import numpy as np
import tensorflow as tf

# Function to load class names from labels.txt
def load_labels(label_file):
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# Load the trained model from Teachable Machine
model = tf.keras.models.load_model('model/keras_Model.h5')

# Load the class names from the labels.txt file
class_names = load_labels('model/labels.txt')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the image for model prediction (resize and normalize)
    img = cv2.resize(frame, (224, 224))  # Resize to model's input size
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    prediction_label = class_names[class_idx]
    
    # Display the resulting frame with prediction
    cv2.putText(frame, f'Prediction: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissors', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()