import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tempfile
import os

# Configure Streamlit
st.set_page_config(layout="wide", page_title="Text Detection")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="armvectores/yolov8n_handwritten_text_detection",
            filename="best.pt"
        )
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_model()

# App title
st.title("Handwritten Text Detection using Huggingface API")

# Show both input options
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
image_url = st.text_input("Or enter image URL", "http://images.cocodataset.org/val2017/000000039769.jpg")

# Determine which source to use (priority to uploaded file)
source = None
if uploaded_file is not None:
    source = ("upload", uploaded_file)
elif image_url:
    source = ("url", image_url)

if st.button("Detect Text") and source and model:
    with st.spinner('Processing...'):
        try:
            # Process either file or URL
            if source[0] == "upload":
                file_bytes = np.asarray(bytearray(source[1].read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                source_type = "Uploaded Image"
            else:
                response = requests.get(source[1])
                img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                source_type = "URL Image"
            
            # Create temp file for prediction
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, img)
            
            # Run detection
            results = model.predict(
                source=temp_path,
                conf=0.5,
                imgsz=640,
                save=False
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Process results
            if len(results) > 0:
                res_plotted = results[0].plot()
                detected_img = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Original Image ({source_type})")
                    st.image(original_img, use_column_width=True)
                
                with col2:
                    st.subheader("Detected Text")
                    st.image(detected_img, use_column_width=True)
                
                # Show detection info
                with st.expander("Detection Details"):
                    st.write(f"Source: {source_type}")
                    st.write(f"Detected elements: {len(results[0].boxes)}")
                    st.write("Confidence scores:", [round(float(x), 2) for x in results[0].boxes.conf])
                    if source[0] == "url":
                        st.write(f"Image URL: {source[1]}")
            else:
                st.warning("No text detected in the image")
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass
elif not model:
    st.warning("Model not loaded - check your internet connection")
elif not source:
    st.info("Please provide an image (upload or URL)")