import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple

logger = logging.getLogger(__name__)

def evaluate_model(model_path: str, test_data: Tuple[np.ndarray, np.ndarray], output_dir: str):
    """
    Loads a trained model and evaluates it against test data, generating plots and reports.
    """
    x_test, y_test = test_data
    
    logger.info(f"Loading trained model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Basic Keras evaluation
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Test Set Metrics -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Generate predictions
    logger.info("Generating predictions for detailed metrics...")
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification Report
    logger.info("\n--- Classification Report ---")
    report = classification_report(y_test, y_pred, digits=4)
    logger.info(f"\n{report}")

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - MNIST Digit Recognition')
    plt.ylabel('Actual Digit')
    plt.xlabel('Predicted Digit')
    
    plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {plot_path}")