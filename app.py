import streamlit as st
from PIL import Image
import numpy as np
from joblib import load 

st.set_page_config(page_title="Pneumonia Prediction",layout='wide')

st.markdown("<h1 style='text-align: center; color: white;'>Prediction of whether a person has Pneumonia </h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
data = st.file_uploader("Upload an XRAY Image",type = ['png','jpeg','jpg'])
if data is not None:
    with col2:
       st.image(data,caption="Xray image of the Patient",width = 380)
    image = Image.open(data)
    img_array = np.array(image)
    img_array = np.resize(img_array,(256,256,3))
    img_array = np.array([img_array])
    model = load('pneumonia_model.pkl')
    result = model.predict(img_array)
    if result[0][0] == 1:
        st.error('The patient has Pneumonia')
    else:
        st.success('The patient Doesn\'t have Pneumonia')
