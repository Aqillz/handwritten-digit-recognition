import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data

def predict_random_image():
    model_path = os.path.join("..", "outputs", "best_mnist_cnn.keras")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run training first.")
        return

    print("Loading saved model...")
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    _, (x_test, y_test) = load_and_preprocess_data()
    
    # Select a random image from the test set
    idx = np.random.randint(0, len(x_test))
    image = x_test[idx]
    true_label = y_test[idx]
    
    # Generate prediction (expand dims to simulate batch size of 1)
    prediction_probs = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(prediction_probs)
    confidence = np.max(prediction_probs) * 100
    
    print(f"\n--- Prediction Results ---")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label} (Confidence: {confidence:.2f}%)")
    
    # Visualize the result
    plt.figure(figsize=(4, 4))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Predicted: {predicted_label} ({confidence:.1f}%) | True: {true_label}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    predict_random_image()