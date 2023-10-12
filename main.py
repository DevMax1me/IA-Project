import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import keras
import numpy as np
import cv2

# load model.ckpt
model = keras.models.load_model("maxime-V10.h5")

st.header("Draw a number")
st.subheader("Draw a number between 0 and 9")

col1, col2 = st.columns(2)

with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0)",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color="#FFF",
        background_color="#000",
        update_streamlit=True,
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (28, 28), interpolation = cv2.INTER_AREA)

        newimg = keras.utils.normalize(resized, axis = 1)

        st.image(newimg, width=300)

        newimg = np.array(newimg).reshape(-1, 28, 28, 1)


if canvas_result.image_data is not None:
    predictions = model.predict(newimg)
    
    st.write("## Prediction: ", np.argmax(predictions))
    st.bar_chart(predictions[0])
