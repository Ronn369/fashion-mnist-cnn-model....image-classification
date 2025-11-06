import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io

st.set_page_config(page_title="FashionMNIST Debugger", layout="wide")

# first i have took the model which i have trained
model = tf.keras.models.load_model("fashion_mnist_cnn_model.h5")

CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

st.title("Debug: Fashion MNIST Classifier")
st.write("This debug view shows exactly what image is sent to the model and the top predictions.")

uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # it will show original image which i have uploaded
    st.subheader("Original uploaded image")
    uploaded_bytes = uploaded_file.read()
    orig_img = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    st.image(orig_img, use_column_width=False, width=200)

    # since the mnist used the greyscale image so here i have converted the image into grey
    img = Image.open(io.BytesIO(uploaded_bytes)).convert('L')   
    img = img.resize((28,28))                                   
    img = ImageOps.invert(img)                                  
    st.subheader("Processed (grayscale 28x28, inverted)")
    st.image(img, width=140)

    # here i have converted the 
    arr = np.array(img).astype('float32') / 255.0               # normalized 0..1
    st.write("Processed array shape:", arr.shape)
    st.write("Processed array min/max/mean:", float(arr.min()), float(arr.max()), float(arr.mean()))

    # Show small view of numeric pixel values (rounded)
    st.write("Pixel sample (5x5 top-left):")
    st.write(np.round(arr[:5,:5], 3).tolist())

    # Reshape and predict
    inp = arr.reshape(1,28,28,1)
    preds = model.predict(inp)[0]   # 1D array of 10 probs

    # Show top-3 predictions with probabilities
    top3_idx = preds.argsort()[-3:][::-1]
    st.subheader("Top-3 predictions (class : probability)")
    for i in top3_idx:
        st.write(f"{CLASS_NAMES[i]} : {preds[i]:.4f}")

    # Show raw preds array for full transparency
    st.write("Full probability vector (class order):")
    st.write(np.round(preds,4).tolist())

    # Save processed image to disk (so we can inspect later if needed)
    img.save("last_processed.png")
    st.write("Saved processed image as last_processed.png in app workspace.")
