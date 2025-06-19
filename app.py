import streamlit as st
import os
import tempfile
from PIL import Image
import torch
from torchvision import transforms
import gdown

from unetutils import (
    create_and_load_unet_model,
    test_single_image_streamlit,
    process_video_streamlit
)

# ==== PAGE CONFIG ==== #
st.set_page_config(
    page_title="Concrete Crack Prediction – UNet++", 
    page_icon="🧠", 
    layout="wide"
)

# ==== CUSTOM CSS WITH BACKGROUND IMAGE AND FLOATING ICONS ==== #
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Be Vietnam Pro', sans-serif;
        background: url('https://www.transparenttextures.com/patterns/cubes.png');
        background-color: #f9fcff;
        color: #222;
    }
    .block-container {
        padding-top: 2rem;
    }
    .file-info {
        background-color: #e1f5fe;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #29b6f6;
        font-size: 18px;
        margin-bottom: 1.5rem;
        color: #01579b;
        font-weight: 500;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.05);
    }
    .footer {
        margin-top: 3rem;
        font-size: 15px;
        text-align: center;
        color: #333;
    }
    h1, h2, h3, .stTitle, .stSubheader {
        color: #0d47a1;
        font-weight: 700;
    }
    .image-caption {
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        margin-top: 8px;
        color: #0d47a1;
    }
    .icon-float {
        position: fixed;
        animation: float 6s ease-in-out infinite;
        z-index: 0;
        opacity: 0.08;
    }
    .icon1 { top: 20px; left: 40px; width: 40px; }
    .icon2 { top: 70%; right: 30px; width: 50px; animation-delay: 2s; }
    .icon3 { bottom: 20px; left: 50%; width: 45px; animation-delay: 4s; }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
        100% { transform: translateY(0px); }
    }
    .uploaded-label .stFileUploader label {
        font-size: 22px !important;
        font-weight: 600;
        color: #0d47a1;
    }
    </style>
    <img src="https://cdn-icons-png.flaticon.com/512/3771/3771522.png" class="icon-float icon1">
    <img src="https://cdn-icons-png.flaticon.com/512/4228/4228727.png" class="icon-float icon2">
    <img src="https://cdn-icons-png.flaticon.com/512/3222/3222800.png" class="icon-float icon3">
""", unsafe_allow_html=True)

# ==== HEADER ==== #
# st.image("https://drive.google.com/uc?export=view&id=1q38YVeS0UzjiIALh9USM7S3vPg7wS04p", width=120)
st.title("🧠 Concrete Crack Prediction with UNet++")
st.subheader("Phân vùng vết nứt bê tông từ ảnh hoặc video")

# ==== LOAD MODEL ==== #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# checkpoint_url = 'https://drive.google.com/uc?id=11OmToI6aOg7ALOAhl5pJ8wmQzJnJRSkW'
checkpoint_url='https://drive.google.com/file/d/15dxFjfu-0kUJ-8_LbC9xml1qwFg6A9ze/view?usp=sharing'
checkpoint_path = 'checkpoint_best.pt'
if not os.path.exists(checkpoint_path):
    with st.spinner('Đang tải model checkpoint...'):
        gdown.download(checkpoint_url, checkpoint_path, quiet=False)

model = create_and_load_unet_model(checkpoint_path, device)

transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ==== UPLOADER ==== #
st.markdown("<div class='uploaded-label'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📁 Chọn ảnh (PNG/JPG) hoặc video (MP4)", type=["png", "jpg", "jpeg", "mp4"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    st.markdown(f"<div class='file-info'>🗂️ <b>Tệp đã chọn:</b> {uploaded_file.name} – {uploaded_file.size / 1024:.1f} KB</div>", unsafe_allow_html=True)

    if uploaded_file.type.startswith('image'):
        with st.spinner("🔍 Đang xử lý ảnh..."):
            image = Image.open(file_path).convert('RGB')
            result_image = test_single_image_streamlit(model, image, transform_img, device)

        st.subheader("🖼️ So sánh ảnh trước và sau phân tích")
        col1, col2 = st.columns([1,1], gap="large")
        with col1:
            st.image(image, use_container_width=True)
            st.markdown("<div class='image-caption'>Ảnh gốc</div>", unsafe_allow_html=True)
        with col2:
            st.image(result_image, use_container_width=True)
            st.markdown("<div class='image-caption'>Kết quả phân vùng</div>", unsafe_allow_html=True)

        os.unlink(file_path)

    elif uploaded_file.type == 'video/mp4':
        st.subheader("📊 So sánh video trước và sau phân tích")
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.video(file_path)
            st.markdown("<div class='image-caption'>Video gốc</div>", unsafe_allow_html=True)

        if st.button("▶️ Bắt đầu xử lý video"):
            with st.spinner("⚙️ Đang xử lý video..."):
                output_path = process_video_streamlit(file_path, model, transform_img, device)
            st.success("✅ Video đã được xử lý xong!")

            with col2:
                st.video(output_path)
                st.markdown("<div class='image-caption'>Video kết quả</div>", unsafe_allow_html=True)
        else:
            with col2:
                st.markdown("<div class='image-caption'>Nhấn nút để xử lý và hiển thị kết quả</div>", unsafe_allow_html=True)

# ==== FOOTER ==== #
st.markdown("<div class='footer'>Developed by <b>Trần Quý Thế – 20THXD1</b></div>", unsafe_allow_html=True)
