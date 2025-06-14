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
        animation: bgMove 15s linear infinite;
    }
    @keyframes bgMove {
        0% { background-position: 0% 0%, 100% 0%, 0% 100%; }
        100% { background-position: 100% 0%, 0% 0%, 100% 100%; }
    }
    .block-container {
        padding: 1rem 2rem;
        margin: auto;
        width: 96%;
        background-color: rgba(255,255,255,0.95);
        border-radius: 16px;
        box-shadow: 0 0 12px rgba(0,0,0,0.08);
    }
    .stButton > button {
        background: linear-gradient(90deg, #06B6D4, #3B82F6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75em 1.6em;
        font-weight: 600;
        font-size: 1.15rem;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        background: #0284c7;
        transform: scale(1.05);
    }
    .hero {
        background: linear-gradient(90deg, #06B6D4, #3B82F6);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
        position: relative;
    }
    .hero::after {
        content: "🕶️";
        font-size: 2.5rem;
        position: absolute;
        animation: spin 3s linear infinite;
        right: 2rem;
        top: 1.5rem;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .signature {
        text-align: right;
        font-size: 1rem;
        font-weight: 600;
        color: #444;
        margin-top: 2rem;
        padding-right: 1rem;
    }
    .stMarkdown, .stFileUploader, .stTextInput, .stSelectbox, .stColumns, .stSpinner, .stSuccess {
        font-size: 1.3rem !important;
        color: #111 !important;
    }
    .stImage, .stVideo {
        text-align: center;
    }
    .stImage img, .stVideo video {
        border-radius: 12px;
        width: 38% !important;
        display: inline-block;
        margin: 0 1%;
    }
    .logo {
        position: absolute;
        top: 10px;
        left: 10px;
        width: 80px;
    }
    .file-details {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0f172a;
        background: rgba(6,182,212,0.15);
        padding: 0.75rem 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
        display: inline-block;
        border: 1px solid rgba(6,182,212,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ======== LOGO KHOA ==========
st.markdown("""
<img class="logo" src="https://drive.google.com/uc?id=1q38YVeS0UzjiIALh9USM7S3vPg7wS04p"/>
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
    st.markdown(f"<div class='file-details'>🗂️ Tệp đã chọn: <strong>{uploaded_file.name}</strong> ({uploaded_file.size / 1024:.1f} KB)</div>", unsafe_allow_html=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    if uploaded_file.type.startswith('image'):
        with st.spinner("🔍 Đang phân tích ảnh..."):
            image = Image.open(file_path).convert('RGB')
            result_image = test_single_image_streamlit(model, image, transform_img, device)
        st.markdown("<div class='stImage'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.image(uploaded_file, caption="📷 Ảnh gốc", use_container_width=True)
        with col2:
            st.image(result_image, caption="🧠 Mặt nạ phân vùng", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        os.unlink(file_path)

    elif uploaded_file.type == 'video/mp4':
        st.markdown("<div class='stVideo'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.video(uploaded_file, format="video/mp4")
        with col2:
            if st.button("▶️ Bắt đầu xử lý video", use_container_width=True):
                with st.spinner("🎞️ Đang xử lý video..."):
                    process_video_streamlit(file_path, model, transform_img, device)
                st.success("✅ Video đã được xử lý xong!")
                os.unlink(file_path)
        st.markdown("</div>", unsafe_allow_html=True)

# ======== SIGNATURE ==========
st.markdown("""
    <div class="signature">
        Developed by Trần Quý Thế - 20THXD1
    </div>
""", unsafe_allow_html=True)
