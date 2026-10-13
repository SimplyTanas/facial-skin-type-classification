"""
Bộ kiểm thử tính năng (Unit & Functional Tests) cho Phase 2: Image Processing Pipeline
Kiểm tra tính hoạt động đồng bộ của:
- src.preprocessing
- src.skin_detection
- src.color_space
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Đảm bảo import được module từ thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.preprocessing import (
    read_image, validate_image_array, bgr_to_rgb, rgb_to_bgr,
    resize_image, apply_gaussian_blur, apply_median_blur,
    apply_histogram_equalization, apply_clahe, normalize_image,
    baseline_preprocessing_pipeline, enhanced_preprocessing_pipeline,
    compare_before_after
)
from src.skin_detection import (
    detect_skin_hsv, detect_skin_ycrcb, create_skin_mask,
    apply_morphology, fill_mask_holes, clean_skin_mask,
    extract_roi, get_skin_pixels, evaluate_mask_quality,
    baseline_skin_detection, enhanced_skin_detection,
    visualize_skin_detection
)
from src.color_space import (
    convert_to_hsv, convert_to_ycrcb, convert_to_gray, convert_to_lab,
    split_channels, visualize_channels, compare_color_spaces,
    extract_color_statistics, visualize_color_distribution
)


def create_synthetic_face_image(width=300, height=300):
    """Tạo một ảnh nhân tạo giả lập khuôn mặt có vùng da và nền để kiểm thử."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Nền xanh xám
    img[:] = (80, 80, 80)
    # Vùng da elip ở giữa: Màu da BGR xấp xỉ (B=140, G=170, R=220)
    center = (width // 2, height // 2)
    axes = (width // 3, height // 3)
    cv2.ellipse(img, center, axes, 0, 0, 360, (140, 170, 220), -1)
    # Vùng mắt (không phải da)
    cv2.circle(img, (center[0] - 40, center[1] - 30), 12, (20, 20, 20), -1)
    cv2.circle(img, (center[0] + 40, center[1] - 30), 12, (20, 20, 20), -1)
    return img


def test_preprocessing():
    print("Testing Preprocessing...")
    img = create_synthetic_face_image()

    # 1. BGR / RGB
    rgb = bgr_to_rgb(img)
    assert rgb.shape == img.shape
    bgr = rgb_to_bgr(rgb)
    assert np.array_equal(img, bgr)

    # 2. Resizing
    resized_direct, meta_direct = resize_image(img, target_size=(224, 224), keep_aspect_ratio=False)
    assert resized_direct.shape[:2] == (224, 224)

    resized_pad, meta_pad = resize_image(img, target_size=(256, 256), keep_aspect_ratio=True)
    assert resized_pad.shape[:2] == (256, 256)

    # 3. Filtering
    blurred, g_info = apply_gaussian_blur(img, kernel_size=(5, 5), sigma=1.0)
    assert blurred.shape == img.shape

    median, m_info = apply_median_blur(img, kernel_size=5)
    assert median.shape == img.shape

    # 4. Histogram Equalization & CLAHE
    eq_img, eq_info = apply_histogram_equalization(img, color_space="YCrCb", is_bgr=True)
    assert eq_img.shape == img.shape

    clahe_img, clahe_info = apply_clahe(img, clip_limit=2.0, color_space="LAB", is_bgr=True)
    assert clahe_img.shape == img.shape

    # 5. Normalization
    norm_img = normalize_image(img)
    assert norm_img.dtype == np.float32
    assert 0.0 <= norm_img.min() and norm_img.max() <= 1.0

    # 6. Pipelines
    base_out, base_meta = baseline_preprocessing_pipeline(img, target_size=(224, 224), normalize=True)
    assert base_out.shape[:2] == (224, 224)

    enh_out, enh_meta = enhanced_preprocessing_pipeline(img, target_size=(224, 224), normalize=True)
    assert enh_out.shape[:2] == (224, 224)

    print(" Preprocessing tests PASSED!")


def test_skin_detection():
    print("Testing Skin Detection...")
    img = create_synthetic_face_image()

    # 1. Color thresholding
    mask_hsv, info_hsv = detect_skin_hsv(img, is_bgr=True)
    assert mask_hsv.ndim == 2

    mask_ycrcb, info_ycrcb = detect_skin_ycrcb(img, is_bgr=True)
    assert mask_ycrcb.ndim == 2

    mask_comb, info_comb = create_skin_mask(img, method="combined", is_bgr=True)
    assert mask_comb.ndim == 2

    # 2. Morphology & Cleaning
    morphed = apply_morphology(mask_comb, operation="open", kernel_size=(5, 5))
    assert morphed.ndim == 2

    cleaned_mask, clean_meta = clean_skin_mask(mask_comb, min_region_area=100, fill_holes=True)
    assert cleaned_mask.ndim == 2

    # 3. ROI & Pixels
    extracted, roi_meta = extract_roi(img, cleaned_mask, crop_to_bbox=False)
    assert extracted.shape == img.shape

    skin_pix = get_skin_pixels(img, cleaned_mask)
    assert skin_pix.ndim == 2 and skin_pix.shape[1] == 3
    assert skin_pix.shape[0] > 0

    # 4. Quality evaluation
    quality = evaluate_mask_quality(cleaned_mask)
    assert "skin_ratio" in quality
    assert quality["is_valid"] is True

    # 5. Pipelines
    _, b_ext, b_meta = baseline_skin_detection(img, color_space="hsv")
    assert b_ext.shape == img.shape

    _, e_ext, e_meta = enhanced_skin_detection(img, method="combined")
    assert e_ext.shape == img.shape

    print(" Skin Detection tests PASSED!")


def test_color_space():
    print("Testing Color Space...")
    img = create_synthetic_face_image()

    # 1. Conversions
    hsv = convert_to_hsv(img, is_bgr=True)
    assert hsv.shape == img.shape

    ycrcb = convert_to_ycrcb(img, is_bgr=True, standard_order=False)
    assert ycrcb.shape == img.shape

    ycbcr = convert_to_ycrcb(img, is_bgr=True, standard_order=True)
    assert ycbcr.shape == img.shape

    gray = convert_to_gray(img, is_bgr=True)
    assert gray.shape == img.shape[:2]

    lab = convert_to_lab(img, is_bgr=True)
    assert lab.shape == img.shape

    # 2. Channels
    channels = split_channels(img, color_space="RGB", is_bgr=True)
    assert len(channels) == 3

    # 3. Statistics
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[100:200, 100:200] = 255
    stats = extract_color_statistics(img, mask=mask, color_space="HSV", is_bgr=True)
    assert len(stats) == 3
    for ch, metrics in stats.items():
        assert "mean" in metrics
        assert "std" in metrics

    print(" Color Space tests PASSED!")


if __name__ == "__main__":
    test_preprocessing()
    test_skin_detection()
    test_color_space()
    print("\nALL PHASE 2 UNIT TESTS COMPLETED SUCCESSFULLY!")
