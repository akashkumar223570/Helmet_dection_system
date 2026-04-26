import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import time

# ========================= CONFIGURATION =========================
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional look
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stApp {background-color: #f8f9fa;}
    h1 {color: #1f77b4;}
    .detection-box {border: 2px solid #1f77b4; border-radius: 10px; padding: 1rem; margin: 1rem 0;}
    </style>
""", unsafe_allow_html=True)

# ========================= SIDEBAR =========================
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown("#### Model Configuration")

source_mode = st.sidebar.radio(
    "Choose Input Source",
    options=["Image", "Live Webcam"],
    horizontal=True
)

# Model path - Change this to your trained model
import gdown
import os
from ultralytics import YOLO

# Sidebar input (you can keep or hardcode)
MODEL_URL = st.sidebar.text_input(
    "Google Drive Model URL",
    value="https://drive.google.com/uc?id=159gW7pS_WqHYcshw11kdF9TknvCLV5zb",
    help="Paste Google Drive direct download link"
)

LOCAL_MODEL_PATH = "best.pt"

@st.cache_resource
def load_model_from_drive(url):
    try:
        # Download only once
        if not os.path.exists(LOCAL_MODEL_PATH):
            with st.spinner("⬇️ Downloading model from Google Drive..."):
                gdown.download(url, LOCAL_MODEL_PATH, quiet=False)

        # Load YOLO model
        model = YOLO(LOCAL_MODEL_PATH)
        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

model = load_model_from_drive(MODEL_URL)


st.sidebar.success("✅ Model loaded")
# Thresholds
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=0.95, 
    value=0.45,
    step=0.01,
    help="Higher value = fewer but more accurate detections"
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold (NMS)", 
    min_value=0.1, 
    max_value=0.95, 
    value=0.45,
    step=0.01,
    help="Non-Maximum Suppression threshold"
)

# Show labels and confidence on boxes
show_labels = st.sidebar.checkbox("Show Class Labels & Confidence", value=True)
line_thickness = st.sidebar.slider("Bounding Box Thickness", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 Upload an image containing workers or riders. "
    "The model will detect whether they are wearing helmets or not."
)
# ========================= MAIN APP =========================
st.markdown("""
<h2 style='color:black;'> <b>🪖Helmet Detection System </b></h2>
<p style='color:black;'>
<b> &emsp;&emsp;&emsp;Professional YOLO-based solution for detecting helmets on riders.<br>
Supports high-resolution images with clear annotations and detailed results.</b>
</p>
""", unsafe_allow_html=True)

if source_mode=="Image":
    # File uploader 
    uploaded_file = st.file_uploader(
        "Upload Rider Image (JPG, PNG, JPEG)", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image containing motorcycle/bicycle riders"
    )

    if uploaded_file is not None:
        # Display original image
        original_image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                "<h3 style='color:#415E72;'> Original Image</h3>",
                unsafe_allow_html=True
            )
            st.image(original_image,width="stretch", caption="Uploaded Image")
        
        # Convert PIL to OpenCV for better processing
        opencv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        
        # Run inference with progress
        with st.spinner("Running YOLO inference... Please wait"):
            start_time = time.time()
            
            results = model.predict(
                source=opencv_image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            inference_time = time.time() - start_time
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        # Convert back to RGB for Streamlit
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        
        with col2:
            st.markdown(
                "<h3 style='color:#1E104E;'>Detection Result</h3>",
                unsafe_allow_html=True
            )

            st.image(annotated_pil, width="stretch", caption="Detected Helmets")
        
        # ========================= RESULTS SUMMARY =========================
        st.markdown(
                "<h3 style='color:#1E104E;'>Detection Result</h3>",
                unsafe_allow_html=True
            )
        
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            st.success(f"✅ **{len(boxes)} object(s)** detected in {inference_time:.2f} seconds")
            
        
            import pandas as pd

            detection_data = []

            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                
                # ✅ use slider value
                if conf < conf_threshold:
                    continue

                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = conf * 100

                # 🎨 label
                if "no" in class_name.lower():
                    label = "🚨 No Helmet"
                else:
                    label = "🪖 Helmet"

                detection_data.append({
                    "ID": i + 1,
                    "Type": label,
                    "Confidence": f"{confidence:.1f}%"
                })

            # sort
            detection_data.sort(
                key=lambda x: float(x["Confidence"][:-1]),
                reverse=True
            )

            df = pd.DataFrame(detection_data)

            st.dataframe(df, width="stretch", hide_index=True)


            helmet_count = 0
            no_helmet_count = 0

            for box in boxes:
                class_name = model.names[int(box.cls[0])].lower()

                if "no" in class_name:
                    no_helmet_count += 1
                else:
                    helmet_count += 1

            col1, col2 = st.columns(2)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div style="
                    padding:15px;
                    border-radius:10px;
                    background-color:#f0fdf4;
                    text-align:center;
                    border:3px solid #d1fae5;
                ">
                    <h4 style="margin:0; color:#065f46;">🪖 Helmet</h4>
                    <h2 style="margin:5px 0; color:#16a34a;">{helmet_count}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="
                    padding:15px;
                    border-radius:10px;
                    background-color:#fef2f2;
                    text-align:center;
                    border:3px solid #fecaca;
                ">
                    <h4 style="margin:0; color:#7f1d1d;">🚨 No Helmet</h4>
                    <h2 style="margin:5px 0; color:#dc2626;">{no_helmet_count}</h2>
                </div>
                """, unsafe_allow_html=True)
                if no_helmet_count > 0:
                    st.error("🚨 WARNING: No Helmet Detected!")
                else:
                    st.success("✅ All riders wearing helmets")

                    
        else:
            st.warning("⚠️ No objects detected. Try lowering confidence or using a clearer image.")    
        
            
    else:
        st.info("""
    👆 **Upload an image to start helmet detection**  
        """)

    # ------------------- LIVE WEBCAM MODE -------------------
# ------------------- LIVE WEBCAM MODE (Simple & Reliable - Recommended for Local) -------------------
elif source_mode == "Live Webcam":

    st.markdown(
        "<h3 style='color:#ff4b4b;'>🔴 Real-Time Helmet Detection</h3>",
        unsafe_allow_html=True
    )

    st.info("Using Browser Webcam (WebRTC) — Works Online & After Deployment")

    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    import av

    # 🎯 Video Processor (Core AI Logic)
    class HelmetProcessor(VideoTransformerBase):

        def __init__(self):
            self.conf = conf_threshold

        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            # YOLO Detection
            results = model(img, conf=self.conf, imgsz=416)

            annotated_frame = results[0].plot()

            # Count helmets
            helmet_count = 0
            no_helmet_count = 0

            for box in results[0].boxes:
                cls_name = model.names[int(box.cls[0])].lower()
                if "helmet" in cls_name and "no" not in cls_name:
                    helmet_count += 1
                else:
                    no_helmet_count += 1

            # Overlay text
            import cv2
            cv2.putText(annotated_frame, f"Helmet: {helmet_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(annotated_frame, f"No Helmet: {no_helmet_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 🚨 Alert
            if no_helmet_count > 0:
                cv2.putText(annotated_frame, "⚠️ NO HELMET!", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # 🎥 Start Webcam
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    webrtc_streamer(
        key="helmet-detection",
        video_processor_factory=HelmetProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )

st.markdown("---")
st.markdown("""
<div style="
    margin-top:40px;
    padding:20px;
    text-align:center;
    border-top:1px solid #e5e7eb;
    color:#006A67;
    font-size:18px;
">
 
Real-time Helmet Detection using YOLOv8 • Built with Streamlit <b>Streamlit</b> for real-time inference
<br>
<a href="https://github.com/akashkumar223570" target="_blank">GitHub</a> • 
<a href="https://www.linkedin.com/in/akash-kumar-5b7a74324/" target="_blank">LinkedIn</a>

<span style="font-size:18px;">
© 2026 | Developed by Akash kumar
</span>

</div>
""", unsafe_allow_html=True)