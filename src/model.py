import logging
import tensorflow as tf
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)

def build_cnn_model(input_shape: tuple = (28, 28, 1), num_classes: int = 10) -> tf.keras.Model:
    """
    Constructs and compiles the CNN architecture for digit recognition.
    
    Args:
        input_shape: Shape of a single image tensor.
        num_classes: Number of output classes (0-9).
        
    Returns:
        A compiled tf.keras.Model.
    """
    logger.info(f"Building CNN model with input shape: {input_shape}")
    
    model = models.Sequential([
        # Feature Extraction Block 1
        # Conv2D learns spatial hierarchies of features using 32 filters.
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'),
        # MaxPooling reduces spatial dimensions (28x28 -> 14x14) to control overfitting and computational cost.
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Feature Extraction Block 2
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)), # (14x14 -> 7x7)
        
        # Classification Head
        # Flatten transforms the 3D tensor output into a 1D vector for the Dense layer.
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        
        # Dropout randomly sets 50% of input units to 0 at each update during training time,
        # which helps prevent the network from memorizing the training data.
        layers.Dropout(0.5),
        
        # Softmax outputs a probability distribution over the 10 classes.
        layers.Dense(num_classes, activation='softmax')
    ])

    logger.info("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model