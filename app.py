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

# ======================= CẤU HÌNH TRANG ==========================
st.set_page_config(page_title="UNet++ Segmentation", page_icon="🧠", layout="centered")

# ======================= STYLE TÙY CHỈNH ==========================
st.markdown("""
    <style>
    body {
        background-color: #f3f4f6;
    }
    .main {
        background-color: #f3f4f6;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
        border-radius: 20px;
        background-color: #ffffff;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }
    h1 {
        color: #1E3A8A;
    }
    .stButton > button {
        background-color: #06B6D4;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0ea5e9;
        color: white;
    }
    .stFileUploader, .stImage, .stVideo {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ======================= HEADER ==========================
st.title("🧠 UNet++ Segmentation Demo")
st.markdown("### 🎯 Phân vùng ảnh & video bằng mô hình học sâu UNet++")
st.write("Chọn **ảnh** hoặc **video** để mô hình thực hiện phân vùng đối tượng.")
st.markdown("---")

# ======================= LOAD MODEL ==========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_url = 'https://drive.google.com/uc?id=11OmToI6aOg7ALOAhl5pJ8wmQzJnJRSkW'
checkpoint_path = 'checkpoint_best.pt'
if not os.path.exists(checkpoint_path):
    with st.spinner('⏬ Đang tải model...'):
        gdown.download(checkpoint_url, checkpoint_path, quiet=False)

model = create_and_load_unet_model(checkpoint_path, device)

transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ======================= TẢI FILE ==========================
uploaded_file = st.file_uploader(
    "📁 Chọn ảnh (PNG/JPG) hoặc video (MP4)",
    type=["png", "jpg", "jpeg", "mp4"]
)

# ======================= XỬ LÝ ẢNH ==========================
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    file_type = uploaded_file.type

    if file_type.startswith('image'):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🖼️ Ảnh gốc")
            st.image(uploaded_file, use_column_width=True)

        with col2:
            with st.spinner("🧠 Đang phân tích ảnh..."):
                image = Image.open(file_path).convert('RGB')
                result_image = test_single_image_streamlit(model, image, transform_img, device)
            st.markdown("#### ✅ Kết quả phân vùng")
            st.image(result_image, use_column_width=True)

        os.unlink(file_path)

    elif file_type == 'video/mp4':
        st.markdown("#### 🎥 Video gốc")
        st.video(uploaded_file)

        if st.button("▶️ Bắt đầu xử lý video", use_container_width=True):
            with st.spinner("⚙️ Đang xử lý video..."):
                process_video_streamlit(file_path, model, transform_img, device)

            st.success("✅ Hoàn tất xử lý video!")
            os.unlink(file_path)

# ======================= CHÂN TRANG ==========================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #9ca3af; font-size: 0.9em;'>
        🚀 Được phát triển bằng <b>PyTorch + Streamlit</b> | Thiết kế bởi ChatGPT ✨
    </div>
    """,
    unsafe_allow_html=True
)
