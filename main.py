import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import keras
import numpy as np

# Create a canvas component
model = keras.models.load_model('model_CNN_V6.h5')

st.header("Draw a number")
st.subheader("Draw a number between 0 and 9")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0)",  # Fixed fill color with some opacity
    stroke_width=15,
    stroke_color="#000",
    background_color="#fff",
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img)
    
    # Invert the image
    img = img.convert("L")

    img = img.resize((28, 28))
    img = np.array(img)
    print(img.shape)
    
    img = img.reshape(1, 28, 28, 1)
    st.image(img, width=150)
    
    prediction = model.predict(img)

    st.write("## Prediction: ", np.argmax(prediction))
    st.bar_chart(prediction[0])
