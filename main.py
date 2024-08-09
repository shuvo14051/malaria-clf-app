import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Set page configuration
st.set_page_config(page_title="Malaria Cell Classification", page_icon="🦠", layout="wide")

# Load your model
model = tf.keras.models.load_model('malaria_cls.keras')

# Sidebar for image upload
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png"])

# Main title and description
st.title('Malaria Cell Classification')
st.write("""
    Upload a cell image to classify it as infected or uninfected. The model is trained on images of malaria-infected and uninfected cells.
""")

# Styling with custom CSS
st.markdown("""
    <style>
        .reportview-container {
            background: #f7f9fc;
            padding: 20px;
        }
        .sidebar .sidebar-content {
            background: #e6e6e6;
        }
        .header {
            font-size: 24px;
            font-weight: bold;
            color: #2b2b2b;
        }
        .subheader {
            font-size: 18px;
            font-weight: bold;
            color: #2b2b2b;
        }
        .result {
            font-size: 30px;
            font-weight: bold;
            color: #009688;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Display sample images
st.markdown("### Sample Images")
sample_col1, sample_col2 = st.columns(2)

with sample_col1:
    st.markdown("**Infected Cell**")
    infected_sample = Image.open('samples/infected/C37BP2_thinF_IMG_20150620_133001a_cell_76.png')  
    st.image(infected_sample, caption='Infected Cell', width=250)

with sample_col2:
    st.markdown("**Uninfected Cell**")
    uninfected_sample = Image.open('samples/uninfected/C5NThinF_IMG_20150609_122006_cell_179.png') 
    st.image(uninfected_sample, caption='Uninfected Cell', width=250)

if uploaded_file is not None:
    st.markdown("### Uploaded image")
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=300)
    st.write("")
    st.write("Classifying...")
    
    # Progress spinner
    with st.spinner('Analyzing the image...'):
        time.sleep(2)
        
        # Preprocess the image and make prediction
        img_array = np.array(image.resize((128, 128)))  # Resize as per model input
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0
        
        prediction = model.predict(img_array)
        
        # Display prediction results
        if prediction[0] < 0.5:
            result = "Infected cell"
        else:
            result = "Uninfected cell"
        
        st.write(f"""
                 <div class='result'>{result}</div>
                 """, 
                 unsafe_allow_html=True)
        
else:
    st.markdown("## Please upload an image to classify.")

# Footer
st.write("")
st.markdown("""
    <footer style='text-align: center; margin-top:100px'>
        <p>Developed by [Md Younus Ahamed].</p>
    </footer>
""", unsafe_allow_html=True)
