import streamlit as st 
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt 

st.write("""
         # Plane/Drones/Bird classification!
         """)
st.subheader("📂 Choose files...")

uploaded_files = st.file_uploader("",
                                 accept_multiple_files='directory',
                                 type=['jpg','jpeg','png'])
if uploaded_files is not None:
    if len(uploaded_files==1): 
        st.image(uploaded_files, width='stretch')
    if len(uploaded_files > 1): 

### Single image prediction 

### Batch Prediction 