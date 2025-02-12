import streamlit as st
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import time

# Page configuration
st.set_page_config(
    page_title="Dental Disease Detection System",
    page_icon="🦷",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load and cache the YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def detect_disease(model, image):
    """Perform disease detection on the image"""
    try:
        results = model(image)
        return results
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        return None

def draw_boxes(image, results):
    """Draw bounding boxes and labels on the image"""
    # Convert PIL Image to RGB mode if it isn't already
    img = image.convert('RGB')
    draw = ImageDraw.Draw(img)
    
    colors = {'cavity': '#FF0000', 'decay': '#00FF00', 'plaque': '#0000FF'}  # RGB colors
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = result.names[int(box.cls[0])]
            color = colors.get(label.lower(), '#00FF00')
            
            # Draw bounding box
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            
            # Add text
            text = f'{label}: {confidence:.2f}'
            # Draw filled rectangle for text background
            text_bbox = draw.textbbox((x1, y1), text)
            draw.rectangle([text_bbox[0], text_bbox[1]-2, text_bbox[2], text_bbox[3]+2], 
                         fill=color)
            # Draw text in white
            draw.text((x1, y1-2), text, fill='#FFFFFF')
    
    return np.array(img)

def main():
    # Sidebar
    st.sidebar.image("iiotengineers_logo.png", use_container_width=True)  # Updated parameter
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Main content
    st.title("🦷 Dental Disease Detection System")
    st.markdown("""
    This system uses advanced AI to detect dental diseases in images.
    Upload a clear image of teeth for analysis.
    """)
    
    # Load model
    model_path = "best.pt"
    model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load the model. Please check the model path and try again.")
        return
    
    # File uploader with additional info
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"],
        help="Please ensure the image is clear and well-lit for best results."
    )
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            col1.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter
            
            if st.button("Detect Diseases", type="primary"):
                with st.spinner("🔍 Analyzing image..."):
                    start_time = time.time()
                    results = detect_disease(model, image)
                    
                    if results is not None:
                        output_image = draw_boxes(image, results)
                        process_time = time.time() - start_time
                        
                        # Display results in a card-like container
                        st.markdown("""
                            <div style='background-color: #f8f9fa; padding: 1.5rem; 
                                        border-radius: 10px; margin: 1rem 0;'>
                                <h3 style='color: #2C3E50; margin-bottom: 1rem;'>Detection Results</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original Image", use_container_width=True)
                        with col2:
                            st.image(output_image, caption="Detection Results", use_container_width=True)
                        
                        # Display detection statistics
                        st.success(f"Processing completed in {process_time:.2f} seconds")
                        
                        # Display detection summary
                        st.subheader("Detection Summary")
                        for result in results:
                            for box in result.boxes:
                                label = result.names[int(box.cls[0])]
                                confidence = float(box.conf[0])
                                if confidence >= confidence_threshold:
                                    st.info(f"Detected {label} with {confidence:.2%} confidence")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try uploading a different image")
    
    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Developed by IIoT Engineers | © 2025</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
