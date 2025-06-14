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

st.set_page_config(page_title="UNet++ AI Segmentation", page_icon="🧠", layout="centered")

# ======== STYLE & HEADER ============
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Be Vietnam Pro', sans-serif;
        background: linear-gradient(to bottom right, #f0f9ff, #e0f2fe);
    }
    .block-container {
        max-width: 900px;
        padding: 1.5rem 2rem;
        background-color: #ffffff;
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.07);
    }
    .stButton > button {
        background: linear-gradient(90deg, #06B6D4, #3B82F6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.4em;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #0284c7;
    }
    .upload-info {
        background: #ecfeff;
        border-left: 5px solid #06B6D4;
        padding: 12px 18px;
        border-radius: 10px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .hero {
        background: linear-gradient(90deg, #06B6D4, #3B82F6);
        color: white;
        padding: 2.2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "\1F916\1F5A5\1F4BB\2728";
        position: absolute;
        font-size: 5rem;
        top: 10px;
        left: 10px;
        opacity: 0.08;
    }
    .hero::after {
        content: "\1F52C\1F9E0\1F5F9";
        position: absolute;
        font-size: 4.5rem;
        bottom: 10px;
        right: 10px;
        opacity: 0.08;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1 style="font-size: 2.8rem; font-weight: 700; margin-bottom: 0.5rem;">
        🧠 UNet++ AI Segmentation
    </h1>
    <p style="font-size: 1.1rem;">Concrete Crack Prediction - Phân vùng ảnh & video thông minh với mạng học sâu UNet++</p>
</div>
""", unsafe_allow_html=True)

# ======== LOAD MODEL ============
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

# ======== FILE UPLOAD ============
uploaded_file = st.file_uploader("📁 Tải ảnh (PNG/JPG) hoặc video (MP4)", type=["png", "jpg", "jpeg", "mp4"])

if uploaded_file is not None:
    file_size_kb = len(uploaded_file.getvalue()) / 1024
    file_info_html = f"""
    <div class="upload-info">
        <b>📂 Tên tệp:</b> {uploaded_file.name} <br>
        <b>🧾 Loại:</b> {uploaded_file.type} <br>
        <b>📦 Kích thước:</b> {file_size_kb:.2f} KB <br>
        <b>⏰ Tải lên lúc:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """
    st.markdown(file_info_html, unsafe_allow_html=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    if uploaded_file.type.startswith('image'):
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Ảnh gốc", use_column_width=True)

        with col2:
            with st.spinner("🔍 Đang phân tích ảnh..."):
                image = Image.open(file_path).convert('RGB')
                result_image = test_single_image_streamlit(model, image, transform_img, device)
            st.image(result_image, caption="🎯 Mặt nạ phân vùng", use_column_width=True)

        os.unlink(file_path)

    elif uploaded_file.type == 'video/mp4':
        st.video(uploaded_file)

        if st.button("▶️ Bắt đầu xử lý video", use_container_width=True):
            with st.spinner("🎞️ Đang xử lý video..."):
                process_video_streamlit(file_path, model, transform_img, device)
            st.success("✅ Video đã được xử lý xong!")
            os.unlink(file_path)
