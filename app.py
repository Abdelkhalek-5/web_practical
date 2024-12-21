import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
st.write("Loading model...")
try:
    model = load_model("skin_model.h5")
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define class labels
CLASS_LABELS = {0: 'Benign', 1: 'Malignant'}

# App UI
st.title("Skin Cancer Classification")
st.write("Upload an image to classify it as Benign or Malignant.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.write("Image uploaded successfully!")
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    st.write("Classifying...")
    image = image.resize((150, 150))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    try:
        prediction = model.predict(image_array)
        class_label = CLASS_LABELS[int(prediction[0][0] > 0.5)]  # Threshold for binary classification
        st.write(f"Prediction: **{class_label}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
