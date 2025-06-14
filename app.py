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

# ========== CẤU HÌNH TRANG =============
st.set_page_config(page_title="UNet++ Segmentation", page_icon="🧠", layout="centered")

# ========== FONT + STYLE + TIÊU ĐỀ LỚN =============
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Be Vietnam Pro', sans-serif;
    }
    .block-container {
        max-width: 900px;
        padding: 1.5rem 2rem;
        background-color: #ffffff;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .stButton > button {
        background-color: #06B6D4;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #0ea5e9;
    }
    .stFileUploader, .stImage, .stVideo {
        border-radius: 10px;
        overflow: hidden;
    }
    .upload-info {
        background-color: #f9fafb;
        border-left: 4px solid #06B6D4;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== HEADER LỚN ĐẦU TRANG =====================
st.markdown("""
    <div style="
        background: linear-gradient(90deg, #06B6D4, #3B82F6);
        padding: 2rem 1rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    ">
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700;">
            🧠 UNet++ AI Segmentation
        </h1>
        <p style="font-size: 1.1rem; margin-top: 0.5rem;">
            Phân vùng ảnh & video bằng mạng học sâu UNet++ – Giao diện hiện đại, tốc độ nhanh
        </p>
    </div>
""", unsafe_allow_html=True)

# ========== GIỚI THIỆU NGẮN ============
st.markdown("### 📌 Ứng dụng phân vùng thông minh")
st.write("Tải ảnh hoặc video để mô hình UNet++ tự động phân tích và tạo mặt nạ đối tượng.")
st.markdown("---")

# ========== TẢI MODEL ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_url = 'https://drive.google.com/uc?id=11OmToI6aOg7ALOAhl5pJ8wmQzJnJRSkW'
checkpoint_path = 'checkpoint_best.pt'
if not os.path.exists(checkpoint_path):
    with st.spinner('⏬ Đang tải model từ Google Drive...'):
        gdown.download(checkpoint_url, checkpoint_path, quiet=False)

model = create_and_load_unet_model(checkpoint_path, device)

transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ========== TẢI FILE NGƯỜI DÙNG ============
uploaded_file = st.file_uploader(
    "📁 Chọn ảnh (PNG/JPG) hoặc video (MP4)",
    type=["png", "jpg", "jpeg", "mp4"]
)

# ========== XỬ LÝ ============
if uploaded_file is not None:
    # Hiện thông tin tệp
    file_size_kb = len(uploaded_file.getvalue()) / 1024
    file_info_html = f"""
    <div class="upload-info">
        <b>📂 Tên tệp:</b> {uploaded_file.name} <br>
        <b>🧾 Loại:</b> {uploaded_file.type} <br>
        <b>📦 Kích thước:</b> {file_size_kb:.2f} KB <br>
        <b>⏰ Thời điểm tải:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """
    st.markdown(file_info_html, unsafe_allow_html=True)

    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    file_type = uploaded_file.type

    if file_type.startswith('image'):
        st.markdown("#### 🖼️ Ảnh gốc và kết quả")
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Ảnh gốc", use_column_width=True)

        with col2:
            with st.spinner("🧠 Đang xử lý ảnh..."):
                image = Image.open(file_path).convert('RGB')
                result_image = test_single_image_streamlit(model, image, transform_img, device)
            st.image(result_image, caption="Kết quả phân vùng", use_column_width=True)

        os.unlink(file_path)

    elif file_type == 'video/mp4':
        st.markdown("#### 🎥 Video gốc")
        st.video(uploaded_file)

        if st.button("▶️ Bắt đầu xử lý video", use_container_width=True):
            with st.spinner("⚙️ Đang xử lý video..."):
                process_video_streamlit(file_path, model, transform_img, device)
            st.success("✅ Hoàn tất xử lý video!")
            os.unlink(file_path)

# ========== CHÂN TRANG ============
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>
        🚀 Phát triển bởi <b>UNet++ Lab</b> | Font: Be Vietnam Pro | Tối ưu giao diện laptop 💻
    </div>
    """,
    unsafe_allow_html=True
)
