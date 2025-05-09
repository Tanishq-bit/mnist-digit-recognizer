import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Load the trained model
model = load_model('models/digit_cnn_model.h5')

# Path to test image
img_path = 'test_samples/digit_7.jpg'

# Read and preprocess image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found at: {img_path}")

# Resize and normalize
img_resized = cv2.resize(img, (28, 28))
img_normalized = img_resized / 255.0
img_reshaped = img_normalized.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img_reshaped)
predicted_digit = np.argmax(prediction)

print(f"Predicted Digit: {predicted_digit}")

# Display image with prediction using matplotlib
plt.figure(figsize=(3, 3))
plt.imshow(img_resized, cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
