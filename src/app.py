import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🧠", layout="centered")

# Load model (cached so it doesn't reload on every drawing stroke)
@st.cache_resource
def load_model():
    # Get the absolute path of the directory where app.py is located (src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the correct path relative to app.py
    model_path = os.path.join(current_dir, "..", "outputs", "best_mnist_cnn.keras")
    return tf.keras.models.load_model(model_path)

model = load_model()

st.title("🧠 Handwritten Digit Recognition")
st.write("Draw a single digit (0-9) in the box below, and the CNN will predict it in real-time.")

# Create two columns: one for drawing, one for the prediction output
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Draw Here")
    # Create a canvas component
    # MNIST images are white digits on a black background. We replicate that here.
    canvas_result = st_canvas(
        fill_color="#000000", 
        stroke_width=20,       
        stroke_color="#FFFFFF",
        background_color="#000000", 
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        return_image_data=True, # Add this line
    )

with col2:
    st.markdown("### Prediction")
    if canvas_result.image_data is not None:
        # The canvas outputs an RGBA image (280, 280, 4)
        img = canvas_result.image_data
        
        # Check if the user has actually drawn something (image is not entirely black)
        if np.any(img[:, :, 0] > 0):
            # 1. Convert RGBA to Grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            
            # 2. Resize from 280x280 down to 28x28 using area interpolation
            img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            
            # 3. Normalize pixel values to [0, 1]
            img_normalized = img_resized / 255.0
            
            # 4. Reshape to (1, 28, 28, 1) to match model input shape
            img_reshaped = np.expand_dims(img_normalized, axis=(0, -1))
            
            # Generate prediction
            prediction_probs = model.predict(img_reshaped, verbose=0)
            predicted_digit = np.argmax(prediction_probs)
            confidence = np.max(prediction_probs) * 100
            
            # Display results
            st.markdown(f"<h1 style='text-align: center; font-size: 72px; color: #4CAF50;'>{predicted_digit}</h1>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Display a probability bar chart
            st.bar_chart(prediction_probs[0])
        else:
            st.info("Waiting for input... Draw a digit on the canvas.")
            