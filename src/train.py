import os
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)

def train_model(model: tf.keras.Model, 
                train_data: Tuple[np.ndarray, np.ndarray], 
                val_data: Tuple[np.ndarray, np.ndarray], 
                output_dir: str,
                epochs: int = 20, 
                batch_size: int = 64) -> tf.keras.callbacks.History:
    """
    Executes the training loop with MLOps best practices (callbacks, artifact saving).
    """
    x_train, y_train = train_data
    
    # Ensure output directory exists for saved models
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_mnist_cnn.keras")

    # Callbacks configuration
    # 1. EarlyStopping: Halts training if validation loss doesn't improve for 3 epochs.
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True,
        verbose=1
    )
    
    # 2. ModelCheckpoint: Saves the best model state to disk automatically.
    model_checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    logger.info(f"Starting training loop for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    logger.info(f"Training completed. Best model saved to {model_path}")
    return history