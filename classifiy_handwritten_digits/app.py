import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('epic_num_reader.h5', custom_objects={'softmax_v2': tf.nn.softmax})

st.title("Predict My Number")
st.write("Lukis satu nombor (0-9) dalam kotak kat bawah ni")

canvas_result = st_canvas(
    fill_color="rgba(255,165,0, 0.3)",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (28,28))

    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape(1, 28, 28)

    if st.button('Teka Nombor Ni!'):
        prediction = model.predict(img_reshaped)
        nombor = np.argmax(prediction)
        confidence = np.max(prediction)

        st.header(f"Aku teka  ni nombor: {nombor}")
        st.write(f"Confidence: {confidence*100:.2f}%")