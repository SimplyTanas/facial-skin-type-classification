"""
Module Tiền Xử Lý Ảnh (Image Preprocessing Module)
Dự án: Phân loại loại da mặt (Xử lý ảnh số - Digital Image Processing)

File: src/preprocessing.py
Mục đích:
- Cung cấp các hàm đọc, chuẩn hóa, lọc nhiễu và tăng cường độ tương phản cho ảnh da mặt.
- Hỗ trợ cả pipeline Baseline (cơ bản) và pipeline Enhanced (nâng cao).
- Trả về kết quả trung gian phục vụ trực quan hóa (visualization) và báo cáo học thuật.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Nhập cấu hình từ src.config (có cơ chế dự phòng nếu chạy độc lập)
try:
    from src.config import (
        CLAHE_CLIP_LIMIT,
        CLAHE_TILE_GRID_SIZE,
        GAUSSIAN_KERNEL_SIZE,
        GAUSSIAN_SIGMA,
        MEDIAN_KERNEL_SIZE,
        NORMALIZE_RANGE,
        TARGET_SIZE,
    )
except ImportError:
    TARGET_SIZE = (224, 224)
    GAUSSIAN_KERNEL_SIZE = (5, 5)
    GAUSSIAN_SIGMA = 1.0
    MEDIAN_KERNEL_SIZE = 5
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    NORMALIZE_RANGE = (0.0, 1.0)


# ==============================================================================
# 1. ĐỌC VÀ KIỂM ĐỊNH ẢNH (IMAGE READING & VALIDATION)
# ==============================================================================

def read_image(
    image_input: Union[str, Path, np.ndarray],
    as_rgb: bool = False
) -> np.ndarray:
    """
    Đọc và kiểm định tính hợp lệ của file ảnh.
    Read and validate image from path or verify existing numpy array.

    Hỗ trợ xử lý đường dẫn chứa ký tự tiếng Việt / Unicode trên Windows thông qua
    kết hợp np.fromfile và cv2.imdecode.

    Args:
        image_input: Đường dẫn đến file ảnh (str, Path) hoặc ndarray ảnh đã đọc.
        as_rgb: Nếu True, chuyển đổi ảnh từ BGR sang RGB. Mặc định là False (BGR).

    Returns:
        np.ndarray: Mảng ảnh uint8 [H, W, C] hoặc [H, W].

    Raises:
        FileNotFoundError: Khi file không tồn tại trên đĩa.
        ValueError: Khi file ảnh bị hỏng hoặc định dạng không hợp lệ.
    """
    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("[Lỗi] Mảng ảnh đầu vào rỗng (empty array).")
        image = image_input.copy()
    else:
        path = Path(image_input)
        if not path.is_file():
            raise FileNotFoundError(f"[Lỗi] Không tìm thấy file ảnh tại đường dẫn: {path}")

        # Đọc qua np.fromfile để tránh lỗi font chữ tiếng Việt/Unicode trên Windows
        try:
            image_bytes = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"[Lỗi] Không thể giải mã file ảnh {path}: {e}")

        if image is None:
            raise ValueError(f"[Lỗi] File không phải định dạng ảnh hợp lệ hoặc bị hỏng: {path}")

    # Chuyển đổi sang RGB nếu được yêu cầu
    if as_rgb and len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def validate_image_array(image: np.ndarray, name: str = "image") -> None:
    """
    Kiểm tra tính toàn vẹn của mảng ảnh ndarray.
    Validate basic properties of image array (type, shape, emptiness).

    Args:
        image: Mảng ảnh cần kiểm tra.
        name: Tên biến để hiển thị trong thông báo lỗi.

    Raises:
        TypeError: Nếu đầu vào không phải là np.ndarray.
        ValueError: Nếu mảng rỗng hoặc có số chiều không phù hợp.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"[Lỗi] {name} phải là kiểu np.ndarray, nhận được: {type(image)}")
    if image.size == 0:
        raise ValueError(f"[Lỗi] {name} là mảng rỗng (size == 0).")
    if image.ndim not in (2, 3):
        raise ValueError(f"[Lỗi] {name} phải có 2 chiều (grayscale) hoặc 3 chiều (color). Hiện có: {image.ndim}")


# ==============================================================================
# 2. CHUYỂN ĐỔI BGR / RGB (COLOR CHANNELS CONVERSION)
# ==============================================================================

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi ảnh từ BGR (chuẩn mặc định OpenCV) sang RGB (chuẩn Matplotlib/PIL).
    Convert image from BGR to RGB color space.

    Args:
        image: Ảnh BGR đầu vào dạng np.ndarray [H, W, 3].

    Returns:
        np.ndarray: Ảnh RGB tương ứng.
    """
    validate_image_array(image, "image")
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return image.copy()
    if image.shape[2] != 3:
        raise ValueError(f"[Lỗi] Ảnh cần có 3 kênh màu để chuyển đổi. Số kênh hiện tại: {image.shape[2]}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi ảnh từ RGB sang BGR.
    Convert image from RGB to BGR color space.

    Args:
        image: Ảnh RGB đầu vào dạng np.ndarray [H, W, 3].

    Returns:
        np.ndarray: Ảnh BGR tương ứng.
    """
    validate_image_array(image, "image")
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return image.copy()
    if image.shape[2] != 3:
        raise ValueError(f"[Lỗi] Ảnh cần có 3 kênh màu để chuyển đổi. Số kênh hiện tại: {image.shape[2]}")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# ==============================================================================
# 3. THAY ĐỔI KÍCH THƯỚC ẢNH (IMAGE RESIZING)
# ==============================================================================

def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE,
    keep_aspect_ratio: bool = False,
    interpolation: int = cv2.INTER_LINEAR,
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Thay đổi kích thước ảnh về target_size (Width, Height).
    Resize image to target_size with option to preserve aspect ratio via letterboxing.

    Args:
        image: Ảnh đầu vào [H, W, C] hoặc [H, W].
        target_size: Kích thước đích dạng tuple (target_width, target_height).
        keep_aspect_ratio: Nếu True, giữ nguyên tỷ lệ khung hình và đệm viền (letterboxing).
                           Nếu False, ép trực tiếp kích thước về target_size.
        interpolation: Thuật toán nội suy OpenCV (mặc định cv2.INTER_LINEAR).
        pad_color: Màu đệm khi giữ tỷ lệ khung hình (mặc định màu đen: (0, 0, 0)).

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Ảnh sau khi thay đổi kích thước.
            - Từ điển chứa siêu dữ liệu: kích thước gốc, kích thước mới, hệ số co dãn, độ đệm.
    """
    validate_image_array(image, "image")
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_size

    if orig_h == 0 or orig_w == 0:
        raise ValueError(f"[Lỗi] Kích thước ảnh gốc không hợp lệ: {orig_w}x{orig_h}")

    metadata: Dict[str, Any] = {
        "original_size": (orig_w, orig_h),
        "target_size": (target_w, target_h),
        "keep_aspect_ratio": keep_aspect_ratio,
        "scale": 1.0,
        "padding": (0, 0, 0, 0),  # (top, bottom, left, right)
    }

    if not keep_aspect_ratio:
        resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        metadata["scale_x"] = target_w / orig_w
        metadata["scale_y"] = target_h / orig_h
        return resized, metadata

    # Trường hợp giữ nguyên tỷ lệ khung hình (Letterboxing / Aspect-preserving resize)
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    if image.ndim == 3:
        padded = cv2.copyMakeBorder(
            scaled_img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color
        )
    else:
        padded = cv2.copyMakeBorder(
            scaled_img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color[0]
        )

    metadata["scale"] = scale
    metadata["scaled_size"] = (new_w, new_h)
    metadata["padding"] = (pad_top, pad_bottom, pad_left, pad_right)

    return padded, metadata


# ==============================================================================
# 4. LỌC NHIỄU KHÔNG GIAN (SPATIAL FILTERING & DENOISING)
# ==============================================================================

def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = GAUSSIAN_KERNEL_SIZE,
    sigma: float = GAUSSIAN_SIGMA
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Áp dụng bộ lọc Gaussian để làm mịn ảnh và loại bỏ nhiễu tần số cao.
    Apply Gaussian blur filter with configurable kernel size and sigma.

    Args:
        image: Ảnh đầu vào [H, W, C] hoặc [H, W].
        kernel_size: Kích thước ma trận lọc (kernel_w, kernel_h), phải là số lẻ dương.
        sigma: Độ lệch chuẩn Gaussian (mặc định 1.0).

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Ảnh sau khi lọc Gaussian.
            - Từ điển chứa thông tin lọc và bản đồ sai phân (residual/noise map).
    """
    validate_image_array(image, "image")
    kw, kh = kernel_size
    if kw <= 0 or kh <= 0 or kw % 2 == 0 or kh % 2 == 0:
        raise ValueError(f"[Lỗi] Kernel Gaussian phải là cặp số lẻ dương, nhận: {kernel_size}")

    blurred = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)
    residual = cv2.absdiff(image, blurred)

    info: Dict[str, Any] = {
        "operation": "gaussian_blur",
        "kernel_size": kernel_size,
        "sigma": sigma,
        "residual_mean": float(np.mean(residual)),
        "residual": residual
    }
    return blurred, info


def apply_median_blur(
    image: np.ndarray,
    kernel_size: int = MEDIAN_KERNEL_SIZE
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Áp dụng bộ lọc trung vị (Median Filter) để loại bỏ nhiễu muối tiêu (salt-and-pepper).
    Bộ lọc phi tuyến tính này bảo toàn tốt các cạnh biên của cấu trúc da mặt.
    Apply Median blur filter with configurable odd kernel size.

    Args:
        image: Ảnh đầu vào [H, W, C] hoặc [H, W].
        kernel_size: Kích thước cửa sổ lọc, phải là số nguyên lẻ dương > 1.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Ảnh sau khi lọc trung vị.
            - Từ điển chứa thông tin lọc và bản đồ sai phân.
    """
    validate_image_array(image, "image")
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError(f"[Lỗi] Kernel Median phải là số nguyên lẻ > 1, nhận: {kernel_size}")

    filtered = cv2.medianBlur(image, kernel_size)
    residual = cv2.absdiff(image, filtered)

    info: Dict[str, Any] = {
        "operation": "median_blur",
        "kernel_size": kernel_size,
        "residual_mean": float(np.mean(residual)),
        "residual": residual
    }
    return filtered, info


# ==============================================================================
# 5. TĂNG CƯỜNG ĐỘ TƯƠNG PHẢN (CONTRAST ENHANCEMENT)
# ==============================================================================

def apply_histogram_equalization(
    image: np.ndarray,
    color_space: str = "YCrCb",
    is_bgr: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cân bằng biểu đồ sắc độ toàn cục (Global Histogram Equalization).
    Áp dụng trên kênh độ sáng (Luminance: Y hoặc V hoặc L) để bảo toàn màu sắc tự nhiên của da,
    tránh làm biến dạng sắc tố màu (Color distortion/Hue shifting).

    Args:
        image: Ảnh đầu vào [H, W, 3] hoặc [H, W].
        color_space: Không gian màu tách kênh độ sáng ('YCrCb', 'HSV', 'LAB').
        is_bgr: True nếu ảnh đầu vào là định dạng BGR, False nếu là RGB.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Ảnh sau khi cân bằng sắc độ.
            - Từ điển chứa kênh độ sáng trước và sau khi xử lý.
    """
    validate_image_array(image, "image")

    # Nếu là ảnh đơn sắc (grayscale)
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        gray = image.squeeze()
        equalized = cv2.equalizeHist(gray)
        info = {
            "color_space": "grayscale",
            "luminance_before": gray,
            "luminance_after": equalized
        }
        return equalized, info

    color_space = color_space.upper()

    if color_space == "YCRCB":
        cvt_to = cv2.COLOR_BGR2YCrCb if is_bgr else cv2.COLOR_RGB2YCrCb
        cvt_back = cv2.COLOR_YCrCb2BGR if is_bgr else cv2.COLOR_YCrCb2RGB
        converted = cv2.cvtColor(image, cvt_to)
        lum_idx = 0  # Kênh Y (Luminance)
    elif color_space == "HSV":
        cvt_to = cv2.COLOR_BGR2HSV if is_bgr else cv2.COLOR_RGB2HSV
        cvt_back = cv2.COLOR_HSV2BGR if is_bgr else cv2.COLOR_HSV2RGB
        converted = cv2.cvtColor(image, cvt_to)
        lum_idx = 2  # Kênh V (Value)
    elif color_space == "LAB":
        cvt_to = cv2.COLOR_BGR2Lab if is_bgr else cv2.COLOR_RGB2Lab
        cvt_back = cv2.COLOR_Lab2BGR if is_bgr else cv2.COLOR_Lab2RGB
        converted = cv2.cvtColor(image, cvt_to)
        lum_idx = 0  # Kênh L (Lightness)
    else:
        raise ValueError(f"[Lỗi] Không gian màu không được hỗ trợ: {color_space}. Chọn 'YCrCb', 'HSV' hoặc 'LAB'.")

    lum_before = converted[:, :, lum_idx].copy()
    converted[:, :, lum_idx] = cv2.equalizeHist(lum_before)
    equalized_image = cv2.cvtColor(converted, cvt_back)

    info = {
        "color_space": color_space,
        "luminance_channel_index": lum_idx,
        "luminance_before": lum_before,
        "luminance_after": converted[:, :, lum_idx],
    }
    return equalized_image, info


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
    color_space: str = "LAB",
    is_bgr: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Áp dụng cân bằng biểu đồ sắc độ thích ứng giới hạn tương phản (CLAHE).
    Contrast Limited Adaptive Histogram Equalization.
    Phương pháp này cải thiện độ tương phản cục bộ của vùng da mà không khuếch đại nhiễu quá mức.

    Args:
        image: Ảnh đầu vào [H, W, 3] hoặc [H, W].
        clip_limit: Ngưỡng giới hạn khuếch đại tương phản (mặc định 2.0).
        tile_grid_size: Kích thước lưới ô lân cận (mặc định (8, 8)).
        color_space: Không gian màu xử lý ('LAB', 'YCrCb', 'HSV'). Mặc định 'LAB'.
        is_bgr: True nếu ảnh đầu vào là BGR, False nếu là RGB.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Ảnh sau khi tăng cường CLAHE.
            - Từ điển chứa kênh độ sáng trước/sau và tham số CLAHE.
    """
    validate_image_array(image, "image")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Ảnh grayscale
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        gray = image.squeeze()
        enhanced = clahe.apply(gray)
        info = {
            "color_space": "grayscale",
            "clip_limit": clip_limit,
            "tile_grid_size": tile_grid_size,
            "luminance_before": gray,
            "luminance_after": enhanced
        }
        return enhanced, info

    color_space = color_space.upper()

    if color_space == "LAB":
        cvt_to = cv2.COLOR_BGR2Lab if is_bgr else cv2.COLOR_RGB2Lab
        cvt_back = cv2.COLOR_Lab2BGR if is_bgr else cv2.COLOR_Lab2RGB
        converted = cv2.cvtColor(image, cvt_to)
        lum_idx = 0  # Kênh L (Lightness)
    elif color_space == "YCRCB":
        cvt_to = cv2.COLOR_BGR2YCrCb if is_bgr else cv2.COLOR_RGB2YCrCb
        cvt_back = cv2.COLOR_YCrCb2BGR if is_bgr else cv2.COLOR_YCrCb2RGB
        converted = cv2.cvtColor(image, cvt_to)
        lum_idx = 0  # Kênh Y (Luma)
    elif color_space == "HSV":
        cvt_to = cv2.COLOR_BGR2HSV if is_bgr else cv2.COLOR_RGB2HSV
        cvt_back = cv2.COLOR_HSV2BGR if is_bgr else cv2.COLOR_HSV2RGB
        converted = cv2.cvtColor(image, cvt_to)
        lum_idx = 2  # Kênh V (Value)
    else:
        raise ValueError(f"[Lỗi] Không gian màu không được hỗ trợ: {color_space}.")

    lum_before = converted[:, :, lum_idx].copy()
    converted[:, :, lum_idx] = clahe.apply(lum_before)
    enhanced_image = cv2.cvtColor(converted, cvt_back)

    info = {
        "color_space": color_space,
        "clip_limit": clip_limit,
        "tile_grid_size": tile_grid_size,
        "luminance_channel_index": lum_idx,
        "luminance_before": lum_before,
        "luminance_after": converted[:, :, lum_idx]
    }
    return enhanced_image, info


def normalize_image(
    image: np.ndarray,
    target_range: Tuple[float, float] = NORMALIZE_RANGE,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Chuẩn hóa giá trị cường độ điểm ảnh về khoảng xác định (thường là [0.0, 1.0]).
    Normalize pixel values to specified range.

    Args:
        image: Ảnh đầu vào kiểu uint8 hoặc float.
        target_range: Khoảng giá trị đích (min_val, max_val).
        dtype: Kiểu dữ liệu đầu ra (mặc định np.float32).

    Returns:
        np.ndarray: Ảnh đã chuẩn hóa dạng float32.
    """
    validate_image_array(image, "image")
    img_float = image.astype(dtype)
    min_val, max_val = target_range

    if image.dtype == np.uint8:
        # Chuẩn hóa từ [0, 255] sang [min_val, max_val]
        norm = (img_float / 255.0) * (max_val - min_val) + min_val
    else:
        current_min = float(np.min(img_float))
        current_max = float(np.max(img_float))
        if current_max - current_min < 1e-7:
            norm = np.zeros_like(img_float) + min_val
        else:
            norm = (img_float - current_min) / (current_max - current_min) * (max_val - min_val) + min_val

    return norm.astype(dtype)


# ==============================================================================
# 6. PIPELINES TIỀN XỬ LÝ (BASELINE & ENHANCED PIPELINES)
# ==============================================================================

def baseline_preprocessing_pipeline(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = TARGET_SIZE,
    normalize: bool = True,
    as_rgb: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pipeline tiền xử lý cơ bản (Baseline Pipeline):
    Quy trình: Đọc ảnh -> Resize trực tiếp về TARGET_SIZE -> Chuẩn hóa [0, 1].

    Args:
        image_input: Đường dẫn ảnh hoặc mảng ảnh np.ndarray.
        target_size: Kích thước chuẩn hóa (width, height).
        normalize: Có chuẩn hóa giá trị điểm ảnh sang [0, 1] hay không.
        as_rgb: Đảm bảo trả về ảnh hệ màu RGB.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Ảnh sau xử lý hoàn thiện.
            - Từ điển chứa các kết quả trung gian:
              {"raw": ..., "resized": ..., "normalized": ...}
    """
    raw_img = read_image(image_input, as_rgb=as_rgb)

    # 1. Resize trực tiếp
    resized_img, resize_meta = resize_image(
        raw_img,
        target_size=target_size,
        keep_aspect_ratio=False
    )

    # 2. Chuẩn hóa pixel
    if normalize:
        processed_img = normalize_image(resized_img, target_range=NORMALIZE_RANGE)
    else:
        processed_img = resized_img.copy()

    intermediates: Dict[str, Any] = {
        "pipeline": "baseline",
        "raw": raw_img,
        "resized": resized_img,
        "normalized": processed_img if normalize else None,
        "resize_meta": resize_meta
    }
    return processed_img, intermediates


def enhanced_preprocessing_pipeline(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = TARGET_SIZE,
    keep_aspect_ratio: bool = False,
    denoise_method: str = "gaussian",
    contrast_method: str = "clahe",
    normalize: bool = True,
    as_rgb: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pipeline tiền xử lý nâng cao (Enhanced Preprocessing Pipeline):
    Quy trình chuẩn trong xử lý ảnh số da mặt:
    1. Đọc và xác thực ảnh (Input validation).
    2. Resize kích thước (Direct hoặc bảo toàn tỷ lệ khung hình).
    3. Lọc khử nhiễu (Gaussian hoặc Median filter).
    4. Cân bằng độ tương phản cục bộ thích ứng (CLAHE) trên kênh độ sáng (L channel).
    5. Chuẩn hóa thang đo [0, 1].

    Args:
        image_input: Đường dẫn ảnh hoặc mảng ảnh np.ndarray.
        target_size: Kích thước đích (width, height).
        keep_aspect_ratio: Giữ tỷ lệ khung hình có padding viền hay không.
        denoise_method: Phương pháp khử nhiễu ('gaussian', 'median', 'none').
        contrast_method: Phương pháp tăng tương phản ('clahe', 'hist_eq', 'none').
        normalize: Chuẩn hóa pixel sang float [0, 1].
        as_rgb: Đảm bảo kênh màu là RGB.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Ảnh sau xử lý hoàn chỉnh.
            - Từ điển chứa kết quả trung gian tại từng công đoạn:
              {"raw": ..., "resized": ..., "denoised": ..., "contrast_enhanced": ..., "final": ...}
    """
    raw_img = read_image(image_input, as_rgb=as_rgb)

    # Bước 1: Thay đổi kích thước
    resized_img, resize_meta = resize_image(
        raw_img,
        target_size=target_size,
        keep_aspect_ratio=keep_aspect_ratio
    )

    # Bước 2: Khử nhiễu làm mịn
    denoise_method = denoise_method.lower()
    if denoise_method == "gaussian":
        denoised_img, denoise_meta = apply_gaussian_blur(
            resized_img,
            kernel_size=GAUSSIAN_KERNEL_SIZE,
            sigma=GAUSSIAN_SIGMA
        )
    elif denoise_method == "median":
        denoised_img, denoise_meta = apply_median_blur(
            resized_img,
            kernel_size=MEDIAN_KERNEL_SIZE
        )
    elif denoise_method == "none":
        denoised_img = resized_img.copy()
        denoise_meta = {"operation": "none"}
    else:
        raise ValueError(f"[Lỗi] Phương pháp khử nhiễu không hợp lệ: {denoise_method}")

    # Bước 3: Tăng cường độ tương phản trên kênh độ sáng (is_bgr=not as_rgb)
    contrast_method = contrast_method.lower()
    if contrast_method == "clahe":
        enhanced_img, contrast_meta = apply_clahe(
            denoised_img,
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=CLAHE_TILE_GRID_SIZE,
            color_space="LAB",
            is_bgr=(not as_rgb)
        )
    elif contrast_method == "hist_eq":
        enhanced_img, contrast_meta = apply_histogram_equalization(
            denoised_img,
            color_space="YCrCb",
            is_bgr=(not as_rgb)
        )
    elif contrast_method == "none":
        enhanced_img = denoised_img.copy()
        contrast_meta = {"operation": "none"}
    else:
        raise ValueError(f"[Lỗi] Phương pháp tương phản không hợp lệ: {contrast_method}")

    # Bước 4: Chuẩn hóa thang đo
    if normalize:
        final_img = normalize_image(enhanced_img, target_range=NORMALIZE_RANGE)
    else:
        final_img = enhanced_img.copy()

    intermediates: Dict[str, Any] = {
        "pipeline": "enhanced",
        "raw": raw_img,
        "resized": resized_img,
        "denoised": denoised_img,
        "contrast_enhanced": enhanced_img,
        "final": final_img,
        "normalized": final_img if normalize else None,
        "resize_meta": resize_meta,
        "denoise_meta": denoise_meta,
        "contrast_meta": contrast_meta,
    }
    return final_img, intermediates


# ==============================================================================
# 7. TRỰC QUAN HÓA SO SÁNH TRƯỚC / SAU (BEFORE / AFTER COMPARISON)
# ==============================================================================

def compare_before_after(
    original: np.ndarray,
    processed: np.ndarray,
    title_before: str = "Gốc (Before)",
    title_after: str = "Đã Xử Lý (After)",
    show_diff: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    So sánh trực quan ảnh trước và sau một thao tác xử lý ảnh.
    Visual comparison between original and processed images with difference map and histograms.

    Args:
        original: Ảnh gốc (RGB hoặc grayscale).
        processed: Ảnh sau xử lý (RGB hoặc grayscale).
        title_before: Tiêu đề ảnh gốc.
        title_after: Tiêu đề ảnh sau xử lý.
        show_diff: Hiển thị bản đồ sai khác nhiệt (difference heatmap).
        save_path: Đường dẫn lưu ảnh biểu đồ (tùy chọn).

    Returns:
        plt.Figure: Đối tượng Figure của Matplotlib.
    """
    validate_image_array(original, "original")
    validate_image_array(processed, "processed")

    # Chuẩn hóa về uint8 [0, 255] để hiển thị đúng nếu ảnh là float [0, 1]
    def _to_display_uint8(img: np.ndarray) -> np.ndarray:
        if np.issubdtype(img.dtype, np.floating):
            clamped = np.clip(img, 0.0, 1.0)
            return (clamped * 255).astype(np.uint8)
        return img.astype(np.uint8)

    disp_orig = _to_display_uint8(original)
    disp_proc = _to_display_uint8(processed)

    # Đảm bảo cùng kích thước để tính sai phân
    if disp_orig.shape != disp_proc.shape:
        disp_orig_resized = cv2.resize(disp_orig, (disp_proc.shape[1], disp_proc.shape[0]))
    else:
        disp_orig_resized = disp_orig

    num_cols = 4 if show_diff else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4.5))

    # Cột 1: Ảnh gốc
    if disp_orig.ndim == 2:
        axes[0].imshow(disp_orig, cmap="gray")
    else:
        axes[0].imshow(disp_orig)
    axes[0].set_title(title_before, fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Cột 2: Ảnh sau xử lý
    if disp_proc.ndim == 2:
        axes[1].imshow(disp_proc, cmap="gray")
    else:
        axes[1].imshow(disp_proc)
    axes[1].set_title(title_after, fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Cột 3: Bản đồ sai phân (Difference Map)
    if show_diff:
        diff = cv2.absdiff(disp_orig_resized, disp_proc)
        if diff.ndim == 3:
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        else:
            diff_gray = diff
        im_diff = axes[2].imshow(diff_gray, cmap="inferno")
        axes[2].set_title("Bản đồ sai khác (|Diff|)", fontsize=12, fontweight="bold")
        axes[2].axis("off")
        fig.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
        hist_ax = axes[3]
    else:
        hist_ax = axes[2]

    # Cột 4 (hoặc 3): Biểu đồ so sánh phân bố Histogram
    if disp_orig_resized.ndim == 3:
        orig_gray = cv2.cvtColor(disp_orig_resized, cv2.COLOR_RGB2GRAY)
    else:
        orig_gray = disp_orig_resized

    if disp_proc.ndim == 3:
        proc_gray = cv2.cvtColor(disp_proc, cv2.COLOR_RGB2GRAY)
    else:
        proc_gray = disp_proc

    hist_ax.hist(orig_gray.ravel(), bins=64, range=(0, 256), color="steelblue", alpha=0.6, label="Trước (Before)")
    hist_ax.hist(proc_gray.ravel(), bins=64, range=(0, 256), color="crimson", alpha=0.6, label="Sau (After)")
    hist_ax.set_title("Biểu đồ sắc độ (Histogram)", fontsize=12, fontweight="bold")
    hist_ax.set_xlim([0, 256])
    hist_ax.legend(loc="upper right", fontsize=9)
    hist_ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200, bbox_inches="tight")

    return fig


__all__ = [
    "read_image",
    "validate_image_array",
    "bgr_to_rgb",
    "rgb_to_bgr",
    "resize_image",
    "apply_gaussian_blur",
    "apply_median_blur",
    "apply_histogram_equalization",
    "apply_clahe",
    "normalize_image",
    "baseline_preprocessing_pipeline",
    "enhanced_preprocessing_pipeline",
    "compare_before_after",
]
