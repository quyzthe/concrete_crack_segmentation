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

# ==== CUSTOM CSS ==== #
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Be Vietnam Pro', sans-serif;
        background: linear-gradient(135deg, #f1f3f9 0%, #e5ecf6 100%);
    }
    .block-container {
        padding-top: 2rem;
    }
    .file-info {
        background-color: #ffffff;
        padding: 0.75rem;
        border-radius: 10px;
        border: 2px dashed #007bff;
        font-size: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #003366;
    }
    .footer {
        margin-top: 3rem;
        font-size: 15px;
        text-align: center;
        color: #444;
    }
    h1, h2, h3, .stTitle, .stSubheader {
        color: #002855;
    }
    </style>
""", unsafe_allow_html=True)

# ==== HEADER ==== #
st.image("https://drive.google.com/uc?export=view&id=1q38YVeS0UzjiIALh9USM7S3vPg7wS04p", width=120)
st.title("🧠 Concrete Crack Prediction with UNet++")
st.subheader("Phân vùng vết nứt bê tông từ ảnh hoặc video")

# ==== LOAD MODEL ==== #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_url = 'https://drive.google.com/uc?id=11OmToI6aOg7ALOAhl5pJ8wmQzJnJRSkW'
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
uploaded_file = st.file_uploader("📁 Chọn ảnh (PNG/JPG) hoặc video (MP4)", type=["png", "jpg", "jpeg", "mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    st.markdown(f"<div class='file-info'>🗂️ <b>Tệp đã chọn:</b> {uploaded_file.name} – {uploaded_file.size / 1024:.1f} KB</div>", unsafe_allow_html=True)

    if uploaded_file.type.startswith('image'):
        with st.spinner("🔍 Đang xử lý ảnh..."):
            image = Image.open(file_path).convert('RGB')
            result_image = test_single_image_streamlit(model, image, transform_img, device)

        st.subheader("📸 Kết quả phân tích ảnh")
        col1, col2 = st.columns([1,1], gap="medium")
        with col1:
            st.image(image, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(result_image, caption="Kết quả phân vùng", use_container_width=True)

        os.unlink(file_path)

    elif uploaded_file.type == 'video/mp4':
        st.subheader("🎥 Kết quả phân tích video")
        col1, col2 = st.columns([1,1], gap="medium")
        with col1:
            st.video(uploaded_file)
        with col2:
            if st.button("▶️ Bắt đầu xử lý video"):
                with st.spinner("⚙️ Đang xử lý video..."):
                    process_video_streamlit(file_path, model, transform_img, device)
                st.success("✅ Video đã được xử lý xong!")
                os.unlink(file_path)

# ==== FOOTER ==== #
st.markdown("<div class='footer'>Developed by <b>Trần Quý Thế – 20THXD1</b></div>", unsafe_allow_html=True)
