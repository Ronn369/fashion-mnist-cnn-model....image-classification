import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("fashion_mnist_cnn_model.h5")

# Class names (same as training)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("Fashion MNIST Image Classifier")
st.write("Upload an image of a clothing item to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    img = image.eval(img, lambda x: 255-x)
  
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    st.image(img, caption=f"Prediction: {class_names[class_index]}", use_column_width=True)
    st.success(f"Predicted: {class_names[class_index]}")
