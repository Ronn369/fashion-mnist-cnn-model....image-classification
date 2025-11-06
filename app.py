import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# title of the page of the streamlit
st.set_page_config(page_title="Fashion MNIST Image Classifier", layout="centered")

# here i have linked the trained model which i have made
model = tf.keras.models.load_model("fashion_mnist_cnn_model.h5")

# Class names which is present in the mnist
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# this is the title of the page of streamlit
st.title("Image Classifier Using Fashion Mnist")
st.write("Upload an image of a clothing item to see what the model predicts!")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=False, width=200)

    # Convert to grayscale, resize, invert, and normalize
    img = image.convert("L")             # grayscale
    img = img.resize((28, 28))           # resize to 28x28
    img = ImageOps.invert(img)           # invert colors (important for Fashion MNIST)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.subheader(f"Predicted: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
