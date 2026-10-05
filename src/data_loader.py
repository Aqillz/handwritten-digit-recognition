import logging
import numpy as np
import tensorflow as tf
from typing import Tuple

logger = logging.getLogger(__name__)

def load_and_preprocess_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Downloads the MNIST dataset, normalizes pixel values, and reshapes
    for Convolutional Neural Network (CNN) ingestion.
    
    Returns:
        (x_train, y_train), (x_test, y_test): Preprocessed numpy arrays.
    """
    logger.info("Downloading and loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    logger.info(f"Original training data shape: {x_train.shape}")
    
    # Normalization: Neural networks train faster and converge better when 
    # input features are scaled to a [0, 1] range rather than [0, 255].
    logger.info("Normalizing image data to range [0, 1]...")
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshaping: A Conv2D layer expects input in the format (batch, height, width, channels).
    # Since MNIST images are grayscale, we add an explicit channel dimension of 1.
    # Original: (60000, 28, 28) -> Reshaped: (60000, 28, 28, 1)
    logger.info("Reshaping arrays to include channel dimension...")
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    logger.info(f"Final training data shape: {x_train.shape}")
    logger.info(f"Final testing data shape: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)