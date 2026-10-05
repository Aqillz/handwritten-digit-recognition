import argparse
import logging
import os
from src.data_loader import load_and_preprocess_data
from src.model import build_cnn_model
from src.train import train_model
from src.evaluate import evaluate_model

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="MNIST CNN Training and Evaluation Pipeline")
    parser.add_argument("--mode", type=str, choices=['train', 'evaluate', 'all'], default='all',
                        help="Pipeline mode to run: 'train', 'evaluate', or 'all'")
    args = parser.parse_args()

    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_mnist_cnn.keras")

    logger.info(f"Starting pipeline in '{args.mode}' mode.")

    # Data is needed for both training and evaluation
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    if args.mode in ['train', 'all']:
        model = build_cnn_model()
        train_model(model, (x_train, y_train), (x_test, y_test), OUTPUT_DIR)

    if args.mode in ['evaluate', 'all']:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Cannot evaluate. Model not found at {MODEL_PATH}")
            return
        evaluate_model(MODEL_PATH, (x_test, y_test), OUTPUT_DIR)

    logger.info("Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()