# 🧠 Handwritten Digit Recognition with CNN (MNIST)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-blueviolet)

An industry-standard implementation of a **Convolutional Neural Network (CNN)** for recognizing handwritten digits from the **MNIST dataset**.

This repository demonstrates best practices in Machine Learning engineering, including:

* Modular code structure
* Proper data ingestion and preprocessing
* CNN-based image classification
* Callback-driven training loops
* Early stopping
* Model checkpointing
* Comprehensive model evaluation
* Classification reports
* Confusion matrix visualization
* Command-line pipeline execution

---

## 📂 Project Architecture

```text
mnist-cnn-pipeline/
│
├── data/
│   └── # Data caching directory
│
├── notebooks/
│   └── # Jupyter notebooks for exploratory data analysis (EDA)
│
├── outputs/
│   ├── # Generated trained models (.keras)
│   └── # Evaluation artifacts (.png)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Data extraction, normalization, and reshaping
│   ├── model.py             # CNN architecture using Keras Sequential API
│   ├── train.py             # Training orchestrator and callbacks
│   └── evaluate.py          # Evaluation, classification reports, and confusion matrix
│
├── main.py                  # CLI pipeline orchestrator
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mnist-cnn-pipeline.git
cd mnist-cnn-pipeline
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Pipeline

The complete workflow is orchestrated through `main.py` using command-line arguments.

### Train and Evaluate

Run the complete pipeline:

```bash
python main.py --mode all
```

### Train Only

To train the CNN model without evaluation:

```bash
python main.py --mode train
```

### Evaluate an Existing Model

To evaluate a previously trained model:

```bash
python main.py --mode evaluate
```

---

## 🏗️ Model Architecture

The network uses a **VGG-style convolutional architecture** adapted for the `28 × 28` grayscale MNIST images.

### Feature Extraction

The feature extraction component consists of two convolutional blocks:

```text
Conv2D
↓
Conv2D
↓
MaxPooling2D
```

The model uses:

* `3 × 3` convolution kernels
* `2 × 2` max-pooling
* ReLU activation functions

The convolutional layers learn spatial features such as:

* Edges
* Curves
* Lines
* Corners
* Digit-specific patterns

Max pooling progressively reduces the spatial dimensions while retaining important features.

### Classification Head

The extracted features are passed through the following classification layers:

```text
Flatten
↓
Dense(128)
↓
Dropout(0.5)
↓
Dense(10, Softmax)
```

The final `Dense(10)` layer represents the ten possible MNIST digit classes:

```text
0 1 2 3 4 5 6 7 8 9
```

A **50% dropout rate** is used to reduce overfitting and encourage the network to learn robust feature representations.

---

## 🧠 Training Strategy

The training pipeline uses TensorFlow/Keras callbacks to improve training reliability.

### Early Stopping

The model monitors:

```text
val_loss
```

Training automatically stops when the validation loss no longer improves.

The best-performing weights are restored after training.

### Model Checkpointing

The best model is saved during training, allowing the pipeline to retain the model with the strongest validation performance rather than simply using the final training epoch.

---

## 📊 Results

The trained CNN achieved:

### **99.30% Test Accuracy**

on the unseen MNIST test dataset.

| Metric        | Performance |
| ------------- | ----------: |
| Test Accuracy |  **99.30%** |
| Precision     |       ~0.99 |
| Recall        |       ~0.99 |
| F1-Score      |       ~0.99 |

Performance was consistently strong across the ten digit classes.

---

## 📈 Training

The training process used **Early Stopping** based on validation loss.

Training stopped at:

```text
Epoch 9
```

with the optimal model weights restored from:

```text
Epoch 6
```

This helps prevent unnecessary training after the model has reached its best validation performance.

---

## 🔍 Model Evaluation

The evaluation pipeline generates several performance artifacts.

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

for each individual digit class.

### Confusion Matrix

A confusion matrix is generated to visualize how accurately the model distinguishes between different handwritten digits.

The generated visualization can be found in:

```text
outputs/
```

---

## 📁 Output Artifacts

After running the pipeline, generated artifacts are stored in the `outputs/` directory.

Example:

```text
outputs/
├── best_model.keras
├── confusion_matrix.png
└── classification_report.txt
```

The exact files may vary depending on the implementation and configuration.

---

## 🛠️ Technologies Used

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| Python           | Core programming language      |
| TensorFlow       | Deep learning framework        |
| Keras            | Neural network API             |
| NumPy            | Numerical computation          |
| Pandas           | Data processing                |
| Scikit-learn     | Model evaluation               |
| Matplotlib       | Visualization                  |
| Seaborn          | Confusion matrix visualization |
| Jupyter Notebook | Exploratory analysis           |

---

## 📦 Dataset

This project uses the **MNIST handwritten digit dataset**.

MNIST contains grayscale images of handwritten digits ranging from `0` to `9`.

Each image has a resolution of:

```text
28 × 28 pixels
```

The images are normalized before being passed into the CNN.

---

## 🔄 Pipeline Workflow

The overall machine learning workflow can be summarized as:

```text
MNIST Dataset
      │
      ▼
Data Loading
      │
      ▼
Normalization
      │
      ▼
Image Reshaping
      │
      ▼
CNN Model Construction
      │
      ▼
Model Training
      │
      ├── Early Stopping
      │
      └── Model Checkpointing
      │
      ▼
Best Model
      │
      ▼
Test Evaluation
      │
      ├── Accuracy
      ├── Precision
      ├── Recall
      ├── F1-Score
      │
      ▼
Confusion Matrix
      │
      ▼
Evaluation Artifacts
```

---

## 💡 Machine Learning Engineering Practices

This project is structured to demonstrate practical Machine Learning Engineering principles rather than keeping the entire workflow inside a single notebook.

Key practices include:

* **Modular architecture** for maintainability
* **Separated data loading and preprocessing**
* **Dedicated model definition**
* **Independent training module**
* **Independent evaluation module**
* **CLI-based pipeline execution**
* **Automated model checkpointing**
* **Early stopping**
* **Reproducible project structure**
* **Generated evaluation artifacts**

This structure makes the project easier to extend into more advanced ML workflows.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

If you would like to contribute:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your changes
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.
