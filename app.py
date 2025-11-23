import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import time
import os
import random

# Import custom YOLOv5 model
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Define the path to your saved YOLOv5 model
model_path = "weights/best.pt"

# Set the device to GPU if available
device = select_device('0' if torch.cuda.is_available() else 'cpu')

# Define a function to draw bounding boxes on the image
def draw_bounding_boxes(image, boxes, confidences, class_ids):
    class_names = ['object']  # Replace with your own class names
    
    # Loop over all the detections
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        class_name = class_names[class_id]
        
        # Draw the bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 128), 2)
        # Optional: Uncomment to show confidence labels
        # label = f"{class_name}: {confidence:.2f}"
        # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)
    
    return image

# Define the Streamlit app
def main():
    # Page configuration
    st.set_page_config(
        page_title="VisionGrid AI",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(120deg, #00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
        }
        .sub-header {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .info-box {
            background-color: #f0f8ff;
            border-left: 5px solid #0072ff;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎯 VisionGrid AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enterprise-Grade Dense Object Detection | Powered by YOLOv5 & SKU-110K</p>', unsafe_allow_html=True)
    
    # Load the model with caching
    @st.cache_resource
    def load_model():
        return attempt_load(model_path, device=device)
    
    try:
        model = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.info("Please ensure model weights are available in the `weights/` directory.")
        model_loaded = False
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Detection Configuration")
        
        st.markdown("### Model Parameters")
        conf_thres = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        iou_thres = st.slider(
            "IOU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Intersection over Union threshold for NMS"
        )
        
        st.markdown("---")
        st.markdown("### 📊 System Information")
        device_info = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🔵 CPU"
        st.info(f"**Compute Device:** {device_info}")
        
        if torch.cuda.is_available():
            st.success(f"**GPU Name:** {torch.cuda.get_device_name(0)}")
        
        st.markdown("---")
        st.markdown("### 📈 Model Statistics")
        st.metric("Baseline mAP@50", "92.2%")
        st.metric("F1-Score", "89.79%")
        st.metric("Precision", "93.0%")
        st.metric("Recall", "86.8%")
    
    # Main content
    if not model_loaded:
        return
    
    st.markdown("### 📤 Upload Image for Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a retail shelf or dense object image"
        )
        
        if uploaded_file is not None:
            st.markdown("#### Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Input Image")
    
    with col2:
        if uploaded_file is not None:
            st.markdown("#### Detection Results")
            
            with st.spinner("🔍 Processing image..."):
                # Start timing
                start_time = time.time()
                
                # Preprocess the image
                original_shape = image.size
                image_resized = image.resize((640, 640))
                img = np.array(image_resized)
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.float() / 255.0
                img = img.unsqueeze(0)

                # Run the YOLOv5 model on the image
                pred = model(img)[0]
                pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)
                
                # Convert to numpy
                pred = [x.detach().cpu().numpy() for x in pred]
                pred = [x.astype(int) for x in pred]
                
                # Post-process the output and draw bounding boxes
                boxes = []
                confidences = []
                class_ids = []
                
                for det in pred:
                    if det is not None and len(det):
                        # Scale the bounding box coordinates to the resized image
                        det[:, :4] = det[:, :4]
                        for *xyxy, conf, cls in det:
                            boxes.append(xyxy)
                            confidences.append(conf.item())
                            class_ids.append(int(cls.item()))
                
                # Draw bounding boxes
                image_with_boxes = np.array(image_resized)
                image_with_boxes = draw_bounding_boxes(image_with_boxes, boxes, confidences, class_ids)
                image_result = Image.fromarray(image_with_boxes)
                
                # Resize back to original dimensions
                image_result = image_result.resize(original_shape)
                
                # Calculate inference time
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Display result
                st.image(image_result, use_column_width=True, caption="Detected Objects")
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Display metrics
            st.markdown("---")
            st.markdown("#### 📊 Detection Metrics")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Objects Detected", len(boxes))
            
            with metric_col2:
                st.metric("Inference Time", f"{inference_time:.1f} ms")
            
            with metric_col3:
                fps = 1000 / inference_time if inference_time > 0 else 0
                st.metric("FPS", f"{fps:.1f}")
            
            if len(confidences) > 0:
                avg_confidence = np.mean(confidences) * 100
                st.progress(avg_confidence / 100)
                st.markdown(f"**Average Confidence:** {avg_confidence:.2f}%")
    
    # Footer information
    st.markdown("---")
    st.markdown("""
        <div class="info-box">
            <h4>🎯 About VisionGrid AI</h4>
            <p>
                VisionGrid AI is a state-of-the-art dense object detection system trained on 110,000+ retail images 
                from the SKU-110K dataset. The model achieves <strong>92.2% mAP@50</strong> with real-time inference 
                capabilities, making it ideal for inventory management, loss prevention, and customer analytics.
            </p>
            <p>
                <strong>Model:</strong> YOLOv5x | <strong>Framework:</strong> PyTorch | <strong>Quantization:</strong> INT8 Support
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Made with ❤️ for the Computer Vision Community | "
        "<a href='https://github.com/naman9271/VisiongridAI'>GitHub Repository</a></p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()