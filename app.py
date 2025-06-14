import streamlit as st
import os
import tempfile
from PIL import Image
import torch
from torchvision import transforms
import gdown
import time

from unetutils import (
    create_and_load_unet_model,
    test_single_image_streamlit,
    process_video_streamlit
)

st.set_page_config(page_title="UNet++ AI Segmentation", page_icon="🧠", layout="wide")

# ======== STYLE & HEADER ==========
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
    html, body {
        font-family: 'Be Vietnam Pro', sans-serif;
        background-color: #f0f4f8;
        background-image: url('https://cdn-icons-png.flaticon.com/512/2202/2202112.png'),
                          url('https://cdn-icons-png.flaticon.com/512/2721/2721273.png'),
                          url('https://cdn-icons-png.flaticon.com/512/2869/2869515.png');
        background-repeat: repeat;
        background-size: 60px;
        background-position: top left, top right, bottom left;
    }
    .block-container {
        padding: 1rem 2rem;
        margin: auto;
        width: 95%;
        background-color: rgba(255,255,255,0.9);
        border-radius: 16px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #06B6D4, #3B82F6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75em 1.6em;
        font-weight: 600;
        font-size: 1.15rem;
    }
    .stButton > button:hover {
        background: #0284c7;
    }
    .hero {
        background: linear-gradient(90deg, #06B6D4, #3B82F6);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
    }
    .hero::after {
        content: "🕶️";
        font-size: 2.5rem;
        position: absolute;
        animation: bounce 1.2s infinite;
        right: 2rem;
        top: 1.5rem;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .stFileUploader, .stImage, .stVideo, .stMarkdown, .stTextInput, .stSelectbox, .stColumns, .stSpinner, .stSuccess {
        font-size: 1.2rem !important;
    }
    img, video {
        border-radius: 12px;
        max-width: 85% !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1 style="font-size: 3.4rem; font-weight: 700; margin-bottom: 0.5rem;">
        🧠 UNet++ AI Segmentation
    </h1>
    <p style="font-size: 1.4rem;">Concrete Crack Prediction - Phân vùng ảnh & video thông minh với mạng học sâu UNet++</p>
</div>
""", unsafe_allow_html=True)

# ======== LOAD MODEL ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_url = 'https://drive.google.com/uc?id=11OmToI6aOg7ALOAhl5pJ8wmQzJnJRSkW'
checkpoint_path = 'checkpoint_best.pt'
if not os.path.exists(checkpoint_path):
    with st.spinner('⏬ Đang tải mô hình...'):
        gdown.download(checkpoint_url, checkpoint_path, quiet=False)

model = create_and_load_unet_model(checkpoint_path, device)
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ======== FILE UPLOAD ==========
uploaded_file = st.file_uploader("📁 Tải ảnh (PNG/JPG) hoặc video (MP4)", type=["png", "jpg", "jpeg", "mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    if uploaded_file.type.startswith('image'):
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Ảnh gốc", use_container_width=False)
        with col2:
            with st.spinner("🔍 Đang phân tích ảnh..."):
                image = Image.open(file_path).convert('RGB')
                result_image = test_single_image_streamlit(model, image, transform_img, device)
            st.image(result_image, caption="🎯 Mặt nạ phân vùng", use_container_width=False)
        os.unlink(file_path)

    elif uploaded_file.type == 'video/mp4':
        st.video(uploaded_file, format="video/mp4")
        if st.button("▶️ Bắt đầu xử lý video", use_container_width=True):
            with st.spinner("🎞️ Đang xử lý video..."):
                process_video_streamlit(file_path, model, transform_img, device)
            st.success("✅ Video đã được xử lý xong!")
            os.unlink(file_path)
