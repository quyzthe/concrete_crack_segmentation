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

st.set_page_config(page_title="UNet++ Segmentation Demo", page_icon="🧠", layout="centered")

st.title("🧠 UNet++ Segmentation Demo")
st.write("Upload ảnh hoặc video để phân vùng đối tượng (Concrete Crack Prediction)")

# ===== LOAD MODEL =====
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

# ===== FILE UPLOADER =====
uploaded_file = st.file_uploader("📁 Chọn file ảnh (PNG/JPG) hoặc video (MP4)", type=["png", "jpg", "jpeg", "mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    st.write(f"**🗂️ Tệp đã chọn:** `{uploaded_file.name}` - {uploaded_file.size / 1024:.1f} KB")

    if uploaded_file.type.startswith('image'):
        st.subheader("📸 Ảnh gốc và kết quả")
        with st.spinner("🔍 Đang xử lý ảnh..."):
            image = Image.open(file_path).convert('RGB')
            result_image = test_single_image_streamlit(model, image, transform_img, device)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(result_image, caption="Kết quả phân vùng", use_container_width=True)

        os.unlink(file_path)

    elif uploaded_file.type == 'video/mp4':
        st.subheader("🎥 Video gốc và kết quả")
        col1, col2 = st.columns(2)
        with col1:
            st.video(uploaded_file)
        with col2:
            if st.button("▶️ Bắt đầu xử lý video"):
                with st.spinner("⚙️ Đang xử lý video..."):
                    process_video_streamlit(file_path, model, transform_img, device)
                st.success("✅ Video đã được xử lý xong!")
                os.unlink(file_path)

st.markdown("---")
st.markdown("**Developed by Trần Quý Thế – 20THXD1**")
