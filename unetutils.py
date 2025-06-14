import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import segmentation_models_pytorch as smp
from skimage.feature import local_binary_pattern

from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip

def create_and_load_unet_model(checkpoint_path, device, encoder_name='resnet34', num_classes=2):
    """
    Tạo mô hình UNet++ và load trọng số từ checkpoint nếu có.
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    ).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model weights from {checkpoint_path}")
    else:
        print("ℹ️ No checkpoint found, using fresh model weights")

    model.eval()
    return model


def get_predicted_mask(model, image, transform, device):
    """
    Trả về mask dự đoán đã resize về kích thước ảnh gốc.
    """
    original_size = image.size  # (width, height)
    image_transformed = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_transformed)
        pred_mask = torch.argmax(output, dim=1)

        pred_mask_resized = F.interpolate(
            pred_mask.unsqueeze(1).float(),
            size=(original_size[1], original_size[0]),
            mode='nearest'
        ).squeeze().cpu().numpy()

    return pred_mask_resized

def merge_and_expand_mask(mask, expand_px=20, connect_dist=5):
    """
    Hậu xử lý mask: mở rộng và nối các vùng gần nhau.

    Parameters:
        mask (np.ndarray): Mask đầu vào (giá trị 0 hoặc 1, dtype=np.uint8).
        expand_px (int): Số pixel mở rộng các vùng mask.
        connect_dist (int): Khoảng cách tối đa giữa các vùng để nối lại.

    Returns:
        np.ndarray: Mask nhị phân sau khi xử lý (giá trị 0 hoặc 255).
    """
    # Đảm bảo mask là nhị phân uint8: 0 hoặc 255
    bin_mask = (mask > 0).astype(np.uint8) * 255

    # Bước 1: Dilation để mở rộng vùng mask
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
    dilated_mask = cv2.dilate(bin_mask, kernel_expand, iterations=1)

    # Bước 2: Closing để nối các vùng gần nhau (morphological closing = dilation + erosion)
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (connect_dist * 2 + 1, connect_dist * 2 + 1))
    connected_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel_connect)

    return connected_mask

def get_bounding_boxes_from_mask(mask, min_area=100, offset=50):
    """
    Trích xuất bounding boxes từ mask nhị phân (pixel = 1).
    Offset mỗi box thêm `offset` pixel về mỗi phía (và giới hạn trong kích thước ảnh).
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = mask.shape
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if w > 0 and h > 0 and area >= min_area:
            x1 = max(x - offset, 0)
            y1 = max(y - offset, 0)
            x2 = min(x + w + offset, width - 1)
            y2 = min(y + h + offset, height - 1)
            boxes.append((x1, y1, x2, y2))

    return boxes



def visualize_prediction_with_boxes(model, image, transform, device):
    """
    Dự đoán mask, vẽ bounding box chồng lên ảnh gốc và hiển thị bằng matplotlib.
    """
    pred_mask_resized = get_predicted_mask(model, image, transform, device)
    boxes = get_bounding_boxes_from_mask(pred_mask_resized)
    # boxes = merge_overlapping_boxes(boxes, threshold=10)
    boxes = boxes

    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    for box in boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor='lime', facecolor='none')
        plt.gca().add_patch(rect)

    plt.imshow(pred_mask_resized, cmap='jet', alpha=0.4)
    plt.title("Dự đoán + Bounding Boxes")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def test_single_image(model, image, transform, device):
    """
    Hiển thị mask dự đoán chồng lên ảnh gốc.
    """
    original_size = image.size  # (width, height)
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1)

        pred_mask_resized = F.interpolate(
            pred_mask.unsqueeze(1).float(),
            size=(original_size[1], original_size[0]),
            mode='nearest'
        ).squeeze().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(pred_mask_resized, cmap='jet', alpha=0.5)
    plt.title("Ảnh gốc + Mask dự đoán")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def test_single_image_streamlit(model, image, transform, device):
    """
    Trả về ảnh mask chồng lên ảnh gốc dưới dạng PIL.Image để hiển thị bằng st.image().
    """
    original_size = image.size  # (width, height)
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1)

        # Resize mask về kích thước ảnh gốc
        pred_mask_resized = F.interpolate(
            pred_mask.unsqueeze(1).float(),
            size=(original_size[1], original_size[0]),
            mode='nearest'
        ).squeeze().cpu().numpy()

    # Tạo ảnh chồng mask bằng matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.imshow(pred_mask_resized, cmap='jet', alpha=0.5)
    ax.axis('off')

    # Lưu vào bộ nhớ đệm
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    # Đọc lại bằng PIL
    result_image = Image.open(buf)
    return result_image


def boxes_overlap(box1, box2, threshold=10):
    """
    Kiểm tra 2 box có overlap hoặc cách nhau không quá threshold pixel.
    Box dạng (x1, y1, x2, y2)
    """
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    # Mở rộng box1 thêm threshold
    x11_ext = x11 - threshold
    y11_ext = y11 - threshold
    x12_ext = x12 + threshold
    y12_ext = y12 + threshold

    # Kiểm tra có overlap (hoặc gần nhau trong threshold) không
    if x12_ext < x21 or x22 < x11_ext:
        return False
    if y12_ext < y21 or y22 < y11_ext:
        return False
    return True


def merge_two_boxes(box1, box2, expand_pixels=50):
    """
    Trả về bounding box bao ngoài cùng của 2 box,
    đồng thời mở rộng thêm expand_pixels pixel ở mỗi cạnh.
    """
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    x1 = min(x11, x21)
    y1 = min(y11, y21)
    x2 = max(x12, x22)
    y2 = max(y12, y22)

    # Mở rộng thêm expand_pixels mỗi cạnh (giữ x1,y1 không âm)
    x1_exp = max(0, x1 - expand_pixels)
    y1_exp = max(0, y1 - expand_pixels)
    x2_exp = x2 + expand_pixels
    y2_exp = y2 + expand_pixels

    return (x1_exp, y1_exp, x2_exp, y2_exp)


def merge_overlapping_boxes(boxes, threshold=50, expand_pixels=0, min_area=500):
    """
    Gộp các bounding boxes chồng lấn hoặc gần nhau thành 1 box duy nhất,
    đồng thời mở rộng mỗi box thêm expand_pixels pixel mỗi cạnh.
    Loại bỏ box có diện tích nhỏ hơn min_area.
    """
    merged = []

    for box in boxes:
        has_merged = False
        for i, mbox in enumerate(merged):
            if boxes_overlap(box, mbox, threshold):
                merged[i] = merge_two_boxes(box, mbox, expand_pixels)
                has_merged = True
                break
        if not has_merged:
            merged.append(box)

    # Lặp lại để hợp nhất tất cả box có thể chạm hoặc gần nhau
    changed = True
    while changed:
        changed = False
        new_merged = []
        while merged:
            box = merged.pop(0)
            i = 0
            while i < len(merged):
                if boxes_overlap(box, merged[i], threshold):
                    box = merge_two_boxes(box, merged[i], expand_pixels)
                    merged.pop(i)
                    changed = True
                else:
                    i += 1
            new_merged.append(box)
        merged = new_merged

    # Lọc loại bỏ box có diện tích < min_area
    filtered = []
    for box in merged:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        if area >= min_area:
            filtered.append(box)

    return filtered



def get_boxes_from_prediction_with_merge(model, image, transform, device):
    """
    Trả về danh sách bounding boxes đã được merge và mask dự đoán.
    """
    mask = get_predicted_mask(model, image, transform, device)
    boxes = get_bounding_boxes_from_mask(mask,900,20)
    merged_boxes = merge_overlapping_boxes(boxes, threshold=0, expand_pixels=0, min_area=1000)
    return merged_boxes, mask



def get_boxes_from_prediction(model, image, transform, device):
    """
    Trả về danh sách bounding boxes đã được merge và mask dự đoán.
    """
    mask = get_predicted_mask(model, image, transform, device)
    boxes = get_bounding_boxes_from_mask(mask,900,0)
    return boxes, mask
    
import cv2
from PIL import Image
import numpy as np
import torch.nn.functional as F
from unetutils import create_and_load_unet_model, get_predicted_mask

import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from unetutils import create_and_load_unet_model, get_predicted_mask

def overlay_mask_on_frame(frame, mask):
    """
    Chồng mask (chỉ những pixel bằng 1) lên frame RGB.
    """
    mask_binary = (mask == 1).astype(np.uint8)
    color_mask = np.zeros_like(frame, dtype=np.uint8)
    color_mask[mask_binary == 1] = [0, 255, 0]  # Màu xanh lá cho crack

    alpha = 0.4
    overlaid = frame.copy()
    overlaid[mask_binary == 1] = (
        alpha * color_mask[mask_binary == 1] +
        (1 - alpha) * frame[mask_binary == 1]
    ).astype(np.uint8)

    return overlaid

def process_video(video_path, model, transform, device, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Không thể mở video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Dùng codec XVID để hỗ trợ .avi
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển frame thành PIL image để dùng transform
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = get_predicted_mask(model, pil_img, transform, device)

        # Resize lại mask về kích thước gốc nếu cần (đã xử lý trong get_predicted_mask)
        overlaid = overlay_mask_on_frame(frame, mask)

        out.write(overlaid)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Đã xử lý {frame_idx} frame")

    out.release()
    cap.release()

    # time.sleep(1) 
    print(f"✅ Video đã lưu tại: {output_path}")


import os
import cv2
import torch
from PIL import Image
import numpy as np
import streamlit as st

import cv2
import os
import time
from PIL import Image
import numpy as np

import cv2
import os
import time
import numpy as np
from PIL import Image

import io
import cv2
from PIL import Image
import numpy as np
import tempfile

def process_video_streamlit(video_path, model, transform, device):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Không thể mở video.")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tạo video tạm trong file để ghi xong rồi đưa vào BytesIO
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_video_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*'H264')  # codec phổ biến
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB → PIL để dùng model
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = get_predicted_mask(model, pil_img, transform, device)

        overlaid = overlay_mask_on_frame(frame, mask)
        overlaid = overlaid.astype(np.uint8)

        # Resize nếu khác kích thước gốc
        if overlaid.shape[:2] != (height, width):
            overlaid = cv2.resize(overlaid, (width, height))

        out.write(overlaid)

    cap.release()
    out.release()

    # Đọc lại video vừa tạo thành BytesIO để dùng luôn
    with open(temp_video_path, "rb") as f:
        video_bytes = f.read()

    video_io = io.BytesIO(video_bytes)
    video_io.seek(0)

    # Hiển thị video ngay
    st.video(temp_video_path)

    # Nút tải video
    st.download_button(
        label="📥 Tải video kết quả",
        data=video_io.getvalue(),
        file_name="processed_video.mp4",
        mime="video/mp4"
    )

    return video_io  # Có thể dùng tiếp nếu cần



def process_video2(video_path, model, transform_img, device, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Không thể mở video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def process_frame(image_pil):
        image_np = np.array(image_pil)
        raw_mask = get_predicted_mask(model, image_pil, transform_img, device)
        mask = merge_and_expand_mask(raw_mask, expand_px=10, connect_dist=100)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_thuhai_aggregated = np.zeros(image_np.shape[:2], dtype=np.uint8)

        for cnt in cnts:
            if cv2.contourArea(cnt) < 100:
                continue

            cnts_expanded = expand_contour(cnt, expand_px=20)
            if len(cnts_expanded) == 0:
                continue
            cnt_exp = max(cnts_expanded, key=cv2.contourArea)

            mask_expanded = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask_expanded, [cnt_exp], -1, 255, thickness=-1)

            cropped_img_masked, bbox = crop_image_by_mask(image_np, mask_expanded)
            if cropped_img_masked is None:
                continue

            cropped_img_pil = Image.fromarray(cropped_img_masked)
            mask_thuhai = get_predicted_mask(model, cropped_img_pil, transform_img, device)

            h_crop, w_crop = mask_thuhai.shape
            x_min, y_min, x_max, y_max = bbox

            mask_expanded_bin = (mask_expanded > 0).astype(np.uint8)
            mask_thuhai_bin = (mask_thuhai > 0).astype(np.uint8)

            mask_thuhai_filtered = mask_thuhai_bin * mask_expanded_bin[y_min:y_min + h_crop, x_min:x_min + w_crop]
            mask_thuhai_filtered = mask_thuhai_filtered * 255

            mask_thuhai_aggregated[y_min:y_min + h_crop, x_min:x_min + w_crop] = np.maximum(
                mask_thuhai_aggregated[y_min:y_min + h_crop, x_min:x_min + w_crop],
                mask_thuhai_filtered
            )

        return image_np, mask_thuhai_aggregated

    def overlay_mask_on_frame(frame_bgr, mask):
        # mask dạng 0 và 255, convert sang 3 channels màu xanh lá
        green_mask = np.zeros_like(frame_bgr, dtype=np.uint8)
        green_mask[:, :, 1] = mask  # kênh G

        alpha = 0.5
        overlaid = cv2.addWeighted(frame_bgr, 1, green_mask, alpha, 0)
        return overlaid

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        image_np, mask_aggregated = process_frame(pil_img)

        overlaid = overlay_mask_on_frame(frame, mask_aggregated)

        out.write(overlaid)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Đã xử lý {frame_idx} frame")

    cap.release()
    out.release()
    print(f"✅ Video đã lưu tại: {output_path}")

def expand_contour(cnt, expand_px=50):
    """
    Mở rộng vùng contour bằng dilation trên mask contour.
    """
    # Tạo mask đủ lớn chứa contour + khoảng mở rộng
    h = cnt[:, 0, 1].max() + expand_px * 2
    w = cnt[:, 0, 0].max() + expand_px * 2
    h = max(h, 1024)
    w = max(w, 1024)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Vẽ contour gốc với offset expand_px để không bị tràn
    cnt_offset = cnt + expand_px
    cv2.drawContours(mask, [cnt_offset], -1, 255, thickness=-1)
    # Dilation mở rộng contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px * 2, expand_px * 2))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    # Tìm contour mới trên mask mở rộng
    cnts_dilated, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Dịch lại contour về vị trí ban đầu (trừ đi offset expand_px)
    cnts_expanded = [c - expand_px for c in cnts_dilated]
    return cnts_expanded

import numpy as np
import cv2

import cv2
import numpy as np

def crop_image_by_mask(image_np, mask, padding=5, min_size=1000, return_3ch=True):
    ys, xs = np.where(mask == 255)
    if len(xs) == 0 or len(ys) == 0:
        return None, None

    h, w = image_np.shape[:2]
    x_min = max(xs.min() - padding, 0)
    x_max = min(xs.max() + padding, w - 1)
    y_min = max(ys.min() - padding, 0)
    y_max = min(ys.max() + padding, h - 1)

    cur_w = x_max - x_min + 1
    cur_h = y_max - y_min + 1
    extra_w = max(min_size - cur_w, 0)
    extra_h = max(min_size - cur_h, 0)

    x_min = max(x_min - extra_w // 2, 0)
    x_max = min(x_max + (extra_w - extra_w // 2), w - 1)
    y_min = max(y_min - extra_h // 2, 0)
    y_max = min(y_max + (extra_h - extra_h // 2), h - 1)

    cropped_img = image_np[y_min:y_max+1, x_min:x_max+1].copy()
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

    # Chuyển ảnh sang không gian màu LAB
    lab = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Áp dụng CLAHE để cải thiện độ tương phản
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    enhanced_lab = cv2.merge((l_clahe, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_uint8 = np.uint8(lbp)  # Chuyển kiểu về uint8 trước khi threshold
    _, lbp_thresh = cv2.threshold(lbp_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lbp_mask = cv2.bitwise_and(cropped_mask, lbp_thresh)

    # Tạo ảnh nền mờ
    blurred_img = cv2.GaussianBlur(cropped_img, (21, 21), sigmaX=10, sigmaY=10)

    # Kết hợp ảnh nền mờ và ảnh vết nứt đã được cải thiện
    result = np.where(lbp_mask[:, :, None] == 255, enhanced_img, blurred_img)

    # Làm mượt kết quả
    result_smooth = cv2.bilateralFilter(result, d=15, sigmaColor=150, sigmaSpace=150)

    # Chuyển ảnh về grayscale
    result_gray = cv2.cvtColor(result_smooth, cv2.COLOR_RGB2GRAY)

    if return_3ch:
        result_out = np.stack([result_gray]*3, axis=-1)
    else:
        result_out = result_gray

    return result_out.astype(np.uint8), (x_min, y_min, x_max, y_max)
