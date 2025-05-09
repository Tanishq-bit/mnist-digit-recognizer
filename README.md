# 🧠 Handwritten Digit Recognition using CNN (MNIST)

This project demonstrates a Convolutional Neural Network (CNN) model built with TensorFlow/Keras to recognize handwritten digits from the MNIST dataset. The trained model is capable of predicting digits from custom grayscale images with high accuracy.

---

## 📌 Project Highlights

- Trains a CNN model on the MNIST dataset
- Recognizes custom handwritten digit images
- Visualizes predictions using OpenCV and Matplotlib
- Cleanly structured and modular Python code

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow 2.x**
- **OpenCV**
- **Matplotlib**
- **NumPy**

---

## 📁 Project Structure

mnist-digit-recognizer/
├── models/
│ └── digit_cnn_model.h5 # Trained model file
├── src/
│ ├── train_model.py # Training script
│ └── predict_digit.py # Prediction script
├── test_samples/
│ └── digit_7.png # test image
├── requirements.txt
└── README.md

Install dependencies

bash
pip install -r requirements.txt
Train the model (or use the pre-trained one in models/)

bash
python src/train_model.py
Run prediction on a sample image

bash
python src/predict_digit.py

🖼️ Sample Prediction Output
<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" width="300" alt="MNIST samples" />