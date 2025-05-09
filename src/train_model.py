from data_loader import load_data
from model_builder import build_model
import os

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    print("Training started...")
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

    os.makedirs("models", exist_ok=True)
    model.save("models/digit_cnn_model.h5")

    _, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
