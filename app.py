import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import io
import os

# Set page configuration
st.set_page_config(
    page_title="Dental Caries Detection",
    page_icon="🦷",
    layout="wide"
)

# Custom CSS to improve app appearance
st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .uploadedFile {
            margin-bottom: 2rem;
        }
        .prediction-text {
            font-size: 1.2rem;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

def load_model():
    """Load the YOLOv5 model"""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
        model.conf = 0.5  # Set confidence threshold
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model):
    """Process the image and return predictions"""
    try:
        # Convert PIL image to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Make prediction
        results = model(image)
        
        # Convert results to DataFrame
        pred_df = results.pandas().xyxy[0]
        
        # Draw boxes on image
        img_array = np.array(image)
        for idx, row in pred_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_array, f"{row['name']} {row['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        return Image.fromarray(img_array), pred_df
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    # App title and description
    st.title("🦷 Dental Caries Detection System")
    st.markdown("""
        This application uses AI to detect dental caries in dental X-ray images.
        Upload an image to get started.
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'best.pt' is in the correct location.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image and display results
        with col2:
            st.subheader("Detected Caries")
            processed_image, predictions = process_image(image, model)
            
            if processed_image is not None:
                st.image(processed_image, caption="Processed Image", use_column_width=True)
                
                if not predictions.empty:
                    st.markdown("### Detection Results")
                    st.markdown("The following caries were detected:")
                    for idx, row in predictions.iterrows():
                        st.markdown(f"- Caries detected with {row['confidence']:.2f} confidence")
                else:
                    st.info("No caries detected in the image.")
            
            # Add download button for processed image
            if processed_image is not None:
                # Convert processed image to bytes
                buf = io.BytesIO()
                processed_image.save(buf, format="PNG")
                btn = st.download_button(
                    label="Download Processed Image",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
