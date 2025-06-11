import streamlit as st
import os
import tempfile
from PIL import Image
import torch
from torchvision import transforms
import time
import gdown

from unetutils import (
    create_and_load_unet_model,
    test_single_image_streamlit,
    process_video_streamlit
)

st.title("🧠 UNet++ Segmentation Demo")
st.write("Upload ảnh hoặc video để phân vùng đối tượng")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Cách 1: Tải checkpoint từ Google Drive ---
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

uploaded_file = st.file_uploader(
    "📁 Chọn file ảnh (PNG/JPG) hoặc video (MP4)",
    type=["png", "jpg", "jpeg", "mp4"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    if uploaded_file.type.startswith('image'):
        st.subheader("📸 Ảnh gốc")
        st.image(uploaded_file, width=400)

        with st.spinner("🔍 Đang xử lý ảnh..."):
            image = Image.open(file_path).convert('RGB')
            result_image = test_single_image_streamlit(model, image, transform_img, device)
            st.subheader("🧠 Kết quả phân vùng")
            st.image(result_image, width=400)

        os.unlink(file_path)

    elif uploaded_file.type == 'video/mp4':
        st.subheader("🎥 Video gốc")
        st.video(uploaded_file)

        # Không nhập thư mục output, tự tạo file temp để lưu kết quả
        output_dir = tempfile.gettempdir()
        output_filename = "processed_video.mp4"
        output_path = os.path.join(output_dir, output_filename)

        if st.button("▶️ Bắt đầu xử lý video"):
            with st.spinner("⚙️ Đang xử lý video..."):
                output_video = process_video_streamlit(file_path, model, transform_img, device, output_path)
                if output_video is not None:
                    st.video(output_video)
                else:
                    st.error("Xử lý video thất bại.")

            os.unlink(file_path)
