import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image,ImageChops
import streamlit_drawable_canvas
from streamlit_drawable_canvas import st_canvas


def load_model():
    #model_path='/Users/yassir2/code/Yassirbenj/amazigh_text/models/amazighmodel3.h5'
    model=tf.keras.models.load_model("interface/amazighmodel3.h5")
    return model

def predict(model,image):
    yhat=model.predict(np.expand_dims(image/255,0))
    labels=['A','B','C','D','DD','AE','F','G',
            'GH','GW','H','HH','J','K','KW','L',
            'M','N','Q','R','RR','S','SS','T',
            'TT','W','X','Y','Z','ZZ','E','I','U']
    proba=np.max(yhat)*100
    y = "{:.2f}".format(proba)
    result=labels[np.argmax(yhat)]
    return result,y

def trim(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    box = diff.getbbox()
    if box:
        image=image.crop(box)
    return image

# Specify canvas parameters in application
#drawing_mode = st.sidebar.selectbox(
   # "Drawing tool:", ("freedraw"))

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
#if drawing_mode == 'point':
   # point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
#stroke_color = st.sidebar.color_picker("Stroke color hex: ")
#bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
#bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

#realtime_update = st.sidebar.checkbox("Update in realtime", True)

with st.form("input_form",clear_on_submit=True):
    st.write("<h3>Upload your image for the magic ✨</h3>", unsafe_allow_html=True)
    canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    #stroke_color=stroke_color,
                    #background_color=bg_color,
                    #background_image=Image.open(bg_image) if bg_image else None,
                    update_streamlit=True,
                    height=250,
                    drawing_mode="freedraw",
                    #point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                    key="canvas",
                    )
    input_img=canvas_result.image_data

    #input_img = st.file_uploader('character image',type=['png', 'jpg','jpeg'])
    if st.form_submit_button("Predict"):
        if input_img is not None:
            img = Image.fromarray(input_img)
            image=trim(img)
            new_image=image.resize((64,64))
            st.image(new_image)
            img_array = np.array(new_image)
            img_array=img_array[:,:,1:4]
            img_tensor=tf.convert_to_tensor(img_array)
            loaded_model = load_model()
            prediction = predict(loaded_model,img_tensor)[0]
            pred_proba= predict(loaded_model,img_tensor)[1]
            if float(pred_proba) < 95:
                st.write(f"<h3>We weren't able to identify this image with sufficient precision, please try again ", unsafe_allow_html=True)
                st.write(f"<h5>The nearest letter is: {prediction} with probability of {pred_proba}% </h3>", unsafe_allow_html=True)
            else:
                st.write(f"<h5>The prediction is: {prediction} with probability of {pred_proba}% </h3>", unsafe_allow_html=True)
