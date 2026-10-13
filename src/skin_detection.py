"""
Module Nhận Diện Vùng Da và Cắt Vùng Quan Tâm (Skin Detection & ROI Module)
Dự án: Phân loại loại da mặt (Xử lý ảnh số - Digital Image Processing)

File: src/skin_detection.py
Mục đích:
- Phân đoạn vùng da mặt bằng ngưỡng không gian màu (HSV, YCrCb/YCbCr, hoặc kết hợp).
- Áp dụng các toán tử hình thái học (Morphology: Erosion, Dilation, Opening, Closing) và lấp lỗ hổng (Hole Filling).
- Trích xuất vùng quan tâm (Region of Interest - ROI) và tập hợp điểm ảnh da (Skin Pixels).
- Đánh giá chất lượng mặt nạ phân đoạn da (Skin Mask Quality Evaluation).
- Cung cấp pipeline Baseline (ngưỡng thuần túy) và Enhanced (kết hợp hình thái học).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Nhập cấu hình từ src.config (có fallback an toàn khi dùng độc lập)
try:
    from src.config import (
        HSV_LOWER,
        HSV_UPPER,
        MIN_SKIN_RATIO_THRESHOLD,
        MIN_SKIN_REGION_AREA,
        MORPH_CLOSE_KERNEL,
        MORPH_KERNEL_SHAPE,
        MORPH_KERNEL_SIZE,
        MORPH_OPEN_KERNEL,
        YCBCR_LOWER,
        YCBCR_UPPER,
        YCRCB_LOWER,
        YCRCB_UPPER,
    )
except ImportError:
    HSV_LOWER = np.array([0, 40, 50], dtype=np.uint8)
    HSV_UPPER = np.array([25, 255, 255], dtype=np.uint8)
    YCRCB_LOWER = np.array([80, 135, 85], dtype=np.uint8)
    YCRCB_UPPER = np.array([255, 180, 135], dtype=np.uint8)
    YCBCR_LOWER = np.array([80, 85, 135], dtype=np.uint8)
    YCBCR_UPPER = np.array([255, 135, 180], dtype=np.uint8)
    MORPH_KERNEL_SIZE = (5, 5)
    MORPH_KERNEL_SHAPE = cv2.MORPH_ELLIPSE
    MORPH_OPEN_KERNEL = (5, 5)
    MORPH_CLOSE_KERNEL = (7, 7)
    MIN_SKIN_REGION_AREA = 150
    MIN_SKIN_RATIO_THRESHOLD = 0.05


# ==============================================================================
# 1. PHÂN ĐOẠN THEO NGƯỠNG KHÔNG GIAN MÀU (COLOR SPACE THRESHOLDING)
# ==============================================================================

def detect_skin_hsv(
    image: np.ndarray,
    lower: Optional[Union[np.ndarray, Tuple[int, int, int]]] = None,
    upper: Optional[Union[np.ndarray, Tuple[int, int, int]]] = None,
    is_bgr: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Phát hiện vùng da người dựa trên ngưỡng màu trong không gian HSV.
    Detect skin region using HSV color space thresholding.

    Ngưỡng khuyến nghị chuẩn cho da người:
    - Hue (Sắc tướng): [0, 25] (trong OpenCV dải H từ 0 - 179)
    - Saturation (Độ bão hòa): [40, 255]
    - Value (Độ sáng): [50, 255]

    Args:
        image: Ảnh đầu vào dạng np.ndarray [H, W, 3].
        lower: Ngưỡng dưới [H, S, V] (mặc định lấy từ config: [0, 40, 50]).
        upper: Ngưỡng trên [H, S, V] (mặc định lấy từ config: [25, 255, 255]).
        is_bgr: True nếu ảnh đầu vào là định dạng BGR, False nếu là RGB.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Mặt nạ nhị phân uint8 [H, W] với giá trị 255 (da) và 0 (nền).
            - Từ điển chứa ảnh HSV và các ngưỡng áp dụng.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("[Lỗi] Mảng ảnh đầu vào không hợp lệ.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"[Lỗi] Ảnh cần có 3 kênh màu, nhận được: {image.shape}")

    lower_np = np.array(lower if lower is not None else HSV_LOWER, dtype=np.uint8)
    upper_np = np.array(upper if upper is not None else HSV_UPPER, dtype=np.uint8)

    # Chuyển đổi sang HSV
    cvt_code = cv2.COLOR_BGR2HSV if is_bgr else cv2.COLOR_RGB2HSV
    hsv_img = cv2.cvtColor(image, cvt_code)

    # Tạo mặt nạ nhị phân
    mask = cv2.inRange(hsv_img, lower_np, upper_np)

    info = {
        "color_space": "HSV",
        "lower_threshold": lower_np.tolist(),
        "upper_threshold": upper_np.tolist(),
        "hsv_image": hsv_img
    }
    return mask, info


def detect_skin_ycrcb(
    image: np.ndarray,
    lower: Optional[Union[np.ndarray, Tuple[int, int, int]]] = None,
    upper: Optional[Union[np.ndarray, Tuple[int, int, int]]] = None,
    is_bgr: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Phát hiện vùng da người dựa trên ngưỡng màu trong không gian YCrCb / YCbCr.
    Detect skin region using YCrCb color space thresholding.

    Trong lý thuyết xử lý ảnh:
    - Y (Luma): [80, 255]
    - Cb (Chroma Blue): [85, 135]
    - Cr (Chroma Red): [135, 180]

    LƯU Ý QUAN TRỌNG VỀ OPENCV:
    OpenCV hàm cv2.cvtColor(..., COLOR_BGR2YCrCb) sắp xếp các kênh theo thứ tự:
    Kênh 0: Y, Kênh 1: Cr, Kênh 2: Cb.
    Hàm này tự động xử lý chuẩn hóa thứ tự kênh để đảm bảo độ chính xác.

    Args:
        image: Ảnh đầu vào dạng np.ndarray [H, W, 3].
        lower: Ngưỡng dưới. Nếu cung cấp theo thứ tự (Y, Cr, Cb) hoặc (Y, Cb, Cr).
        upper: Ngưỡng trên.
        is_bgr: True nếu ảnh đầu vào là BGR, False nếu là RGB.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Mặt nạ nhị phân uint8 [H, W].
            - Từ điển chứa ảnh YCrCb và ngưỡng đã áp dụng.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("[Lỗi] Mảng ảnh đầu vào không hợp lệ.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"[Lỗi] Ảnh cần có 3 kênh màu, nhận được: {image.shape}")

    # Mặc định sử dụng ngưỡng YCrCb của OpenCV: Y in [80, 255], Cr in [135, 180], Cb in [85, 135]
    lower_np = np.array(lower if lower is not None else YCRCB_LOWER, dtype=np.uint8)
    upper_np = np.array(upper if upper is not None else YCRCB_UPPER, dtype=np.uint8)

    # Chuyển đổi sang YCrCb
    cvt_code = cv2.COLOR_BGR2YCrCb if is_bgr else cv2.COLOR_RGB2YCrCb
    ycrcb_img = cv2.cvtColor(image, cvt_code)

    mask = cv2.inRange(ycrcb_img, lower_np, upper_np)

    info = {
        "color_space": "YCrCb",
        "channel_order": "Y-Cr-Cb",
        "lower_threshold": lower_np.tolist(),
        "upper_threshold": upper_np.tolist(),
        "ycrcb_image": ycrcb_img
    }
    return mask, info


def create_skin_mask(
    image: np.ndarray,
    method: str = "combined",
    hsv_lower: Optional[np.ndarray] = None,
    hsv_upper: Optional[np.ndarray] = None,
    ycrcb_lower: Optional[np.ndarray] = None,
    ycrcb_upper: Optional[np.ndarray] = None,
    is_bgr: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Tạo mặt nạ da nhị phân bằng phương pháp lựa chọn (HSV, YCrCb, hoặc kết hợp cả 2).
    Create binary skin mask using chosen method ('hsv', 'ycrcb', 'combined', or 'union').

    Phương pháp 'combined' (giao điểm bitwise AND) kết hợp điểm mạnh của cả HSV
    (loại bỏ ánh sáng yếu/chói) và YCrCb (độc lập với độ sáng, gom cụm sắc tố da rất chặt).

    Args:
        image: Ảnh đầu vào [H, W, 3].
        method: 'combined' (AND), 'union' (OR), 'hsv', hoặc 'ycrcb'.
        hsv_lower, hsv_upper: Ngưỡng HSV tùy chỉnh.
        ycrcb_lower, ycrcb_upper: Ngưỡng YCrCb tùy chỉnh.
        is_bgr: Định dạng ảnh đầu vào là BGR hay RGB.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Mặt nạ nhị phân [H, W] (uint8 0/255).
            - Từ điển chứa các mặt nạ thành phần để minh họa.
    """
    method = method.lower()
    mask_hsv, info_hsv = detect_skin_hsv(image, lower=hsv_lower, upper=hsv_upper, is_bgr=is_bgr)
    mask_ycrcb, info_ycrcb = detect_skin_ycrcb(image, lower=ycrcb_lower, upper=ycrcb_upper, is_bgr=is_bgr)

    if method == "hsv":
        final_mask = mask_hsv
    elif method in ("ycrcb", "ycbcr"):
        final_mask = mask_ycrcb
    elif method == "combined":
        # Phép giao logic: Cả hai điều kiện đều phải thỏa mãn (chính xác cao nhất)
        final_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    elif method == "union":
        # Phép hợp logic: Một trong hai thỏa mãn
        final_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    else:
        raise ValueError(f"[Lỗi] Phương pháp phân đoạn da không hợp lệ: {method}. "
                         f"Hỗ trợ: 'combined', 'union', 'hsv', 'ycrcb'.")

    intermediates = {
        "method": method,
        "mask_hsv": mask_hsv,
        "mask_ycrcb": mask_ycrcb,
        "final_raw_mask": final_mask,
        "info_hsv": info_hsv,
        "info_ycrcb": info_ycrcb
    }
    return final_mask, intermediates


# ==============================================================================
# 2. TOÁN TỬ HÌNH THÁI HỌC VÀ LÀM SẠCH MẶT NẠ (MORPHOLOGY & CLEANING)
# ==============================================================================

def apply_morphology(
    mask: np.ndarray,
    operation: str = "open",
    kernel_size: Tuple[int, int] = MORPH_KERNEL_SIZE,
    kernel_shape: int = MORPH_KERNEL_SHAPE,
    iterations: int = 1
) -> np.ndarray:
    """
    Thực hiện phép biến đổi hình thái học (Morphological Transformation) trên mặt nạ nhị phân.
    Apply morphological operation (erosion, dilation, open, close, gradient) on binary mask.

    Args:
        mask: Mặt nạ nhị phân đầu vào [H, W] (uint8).
        operation: 'open' (mở), 'close' (đóng), 'erode' (co), 'dilate' (giãn), 'gradient'.
        kernel_size: Kích thước phần tử cấu trúc tử (Structuring Element).
        kernel_shape: Hình dạng kernel (cv2.MORPH_ELLIPSE, cv2.MORPH_RECT, cv2.MORPH_CROSS).
        iterations: Số lần lặp lại toán tử.

    Returns:
        np.ndarray: Mặt nạ nhị phân sau khi biến đổi hình thái học.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("[Lỗi] Mặt nạ phải là mảng 2 chiều np.ndarray [H, W].")

    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    op_lower = operation.lower()

    if op_lower in ("open", "opening"):
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op_lower in ("close", "closing"):
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif op_lower in ("erode", "erosion"):
        return cv2.erode(mask, kernel, iterations=iterations)
    elif op_lower in ("dilate", "dilation"):
        return cv2.dilate(mask, kernel, iterations=iterations)
    elif op_lower in ("gradient", "morph_gradient"):
        return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    else:
        raise ValueError(f"[Lỗi] Toán tử hình thái học không hỗ trợ: {operation}")


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Lấp đầy các lỗ hổng bên trong các vùng nhị phân (Fill interior holes/cavities).
    Sử dụng kỹ thuật FloodFill từ góc ngoài hoặc tìm đường viền đóng kín (Contours).

    Args:
        mask: Mặt nạ nhị phân đầu vào [H, W] (0 hoặc 255).

    Returns:
        np.ndarray: Mặt nạ nhị phân đã được lấp kín các lỗ hổng bên trong.
    """
    if mask.ndim != 2:
        raise ValueError("[Lỗi] Mặt nạ phải có 2 chiều [H, W].")

    # Bản sao nhị phân chuẩn 0/255
    bin_mask = (mask > 0).astype(np.uint8) * 255

    # Tạo mảng đệm cho cv2.floodFill (kích thước [H+2, W+2])
    h, w = bin_mask.shape
    floodfill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    im_floodfill = bin_mask.copy()

    # Lan truyền từ điểm (0, 0) để tô đầy vùng nền bên ngoài
    cv2.floodFill(im_floodfill, floodfill_mask, (0, 0), 255)

    # Đảo ngược vùng nền để thu được các lỗ hổng bên trong
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Hợp mặt nạ ban đầu với các lỗ hổng đã tìm thấy
    filled = cv2.bitwise_or(bin_mask, im_floodfill_inv)
    return filled


def clean_skin_mask(
    mask: np.ndarray,
    min_region_area: int = MIN_SKIN_REGION_AREA,
    fill_holes: bool = True,
    open_kernel_size: Tuple[int, int] = MORPH_OPEN_KERNEL,
    close_kernel_size: Tuple[int, int] = MORPH_CLOSE_KERNEL
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Làm sạch mặt nạ da mặt: loại bỏ nhiễu đốm nhỏ, nối liền các mảng da đứt đoạn và lấp lỗ hổng.
    Clean skin mask: remove small isolated noise regions, bridge gaps, and fill holes.

    Quy trình:
    1. Opening: Khử các chấm nhỏ li ti không phải da và tách các cầu nối mỏng.
    2. Closing: Hàn gắn các vết nứt nhỏ và làm mịn đường bao ngoài.
    3. Fill Holes: Lấp các lỗ rỗng bên trong vùng mặt (ví dụ đốm bóng nhờn, sẹo nhỏ).
    4. Connected Components Filter: Loại bỏ các thành phần liên thông có diện tích < min_region_area.

    Args:
        mask: Mặt nạ nhị phân thô đầu vào [H, W].
        min_region_area: Diện tích tối thiểu (pixel) để giữ lại một mảng liên thông.
        fill_holes: Có lấp lỗ hổng bên trong hay không.
        open_kernel_size: Kích thước kernel cho phép mở (Opening).
        close_kernel_size: Kích thước kernel cho phép đóng (Closing).

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Mặt nạ nhị phân đã làm sạch [H, W].
            - Từ điển chứa các mặt nạ tại từng bước trung gian.
    """
    if mask.ndim != 2:
        raise ValueError("[Lỗi] Mặt nạ cần là ảnh 2 chiều [H, W].")

    # 1. Phép mở (Opening) loại bỏ nhiễu hạt
    opened = apply_morphology(
        mask,
        operation="open",
        kernel_size=open_kernel_size,
        kernel_shape=cv2.MORPH_ELLIPSE
    )

    # 2. Phép đóng (Closing) hàn gắn các đứt đoạn
    closed = apply_morphology(
        opened,
        operation="close",
        kernel_size=close_kernel_size,
        kernel_shape=cv2.MORPH_ELLIPSE
    )

    # 3. Lấp lỗ rỗng (Fill holes)
    if fill_holes:
        hole_filled = fill_mask_holes(closed)
    else:
        hole_filled = closed.copy()

    # 4. Lọc thành phần liên thông (Connected Component Analysis)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        hole_filled, connectivity=8
    )

    cleaned = np.zeros_like(hole_filled, dtype=np.uint8)
    retained_components = 0

    # Duyệt qua các thành phần (bỏ qua nhãn 0 là background)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_region_area:
            cleaned[labels == i] = 255
            retained_components += 1

    intermediates = {
        "raw_mask": mask,
        "opened_mask": opened,
        "closed_mask": closed,
        "hole_filled_mask": hole_filled,
        "cleaned_mask": cleaned,
        "total_components_detected": num_labels - 1,
        "components_retained": retained_components,
    }
    return cleaned, intermediates


# ==============================================================================
# 3. TRÍCH XUẤT VÙNG QUAN TÂM (ROI & SKIN PIXELS EXTRACTION)
# ==============================================================================

def extract_roi(
    image: np.ndarray,
    mask: np.ndarray,
    crop_to_bbox: bool = False,
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Trích xuất vùng da mặt từ ảnh ban đầu bằng cách áp dụng mặt nạ nhị phân.
    Extract Region of Interest (ROI) from image using binary skin mask.

    Args:
        image: Ảnh gốc [H, W, C] hoặc [H, W].
        mask: Mặt nạ nhị phân [H, W] (0 hoặc 255).
        crop_to_bbox: Nếu True, cắt khung ảnh khít theo hình chữ nhật bao quanh (Bounding Box).
        background_color: Màu nền thay thế cho các vùng không phải da (mặc định đen).

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Ảnh vùng da đã được tách biệt.
            - Từ điển chứa thông tin bounding box: (x, y, w, h), diện tích, và mặt nạ cắt.
    """
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"[Lỗi] Kích thước ảnh {image.shape[:2]} và mặt nạ {mask.shape[:2]} không khớp.")

    bin_mask = (mask > 0).astype(np.uint8)

    # Áp dụng mặt nạ
    if image.ndim == 3:
        mask_3d = np.repeat(bin_mask[:, :, np.newaxis], image.shape[2], axis=2)
        bg = np.full_like(image, background_color, dtype=image.dtype)
        extracted = np.where(mask_3d == 1, image, bg)
    else:
        bg = np.full_like(image, background_color[0], dtype=image.dtype)
        extracted = np.where(bin_mask == 1, image, bg)

    metadata: Dict[str, Any] = {
        "crop_to_bbox": crop_to_bbox,
        "has_skin": bool(np.any(bin_mask > 0))
    }

    # Cắt gọn theo Bounding Box nếu được yêu cầu
    if crop_to_bbox and np.any(bin_mask > 0):
        x, y, w, h = cv2.boundingRect(mask)
        cropped_extracted = extracted[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        metadata["bbox"] = (x, y, w, h)
        metadata["cropped_mask"] = cropped_mask
        return cropped_extracted, metadata
    else:
        metadata["bbox"] = (0, 0, image.shape[1], image.shape[0])
        metadata["cropped_mask"] = mask
        return extracted, metadata


def get_skin_pixels(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Trích xuất danh sách tất cả các điểm ảnh (pixels) thuộc vùng da.
    Get flat 2D array of skin pixels [N, Channels] where mask > 0.
    Rất hữu ích cho việc tính toán phân bố màu, biểu đồ histogram hoặc đặc trưng độ nhờn/khô.

    Args:
        image: Ảnh gốc [H, W, C] hoặc [H, W].
        mask: Mặt nạ nhị phân [H, W].

    Returns:
        np.ndarray: Mảng 2D kích thước [N, C] chứa các pixel da.
                    Nếu không tìm thấy pixel da nào, trả về mảng rỗng [0, C].
    """
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("[Lỗi] Kích thước ảnh và mặt nạ không tương thích.")

    indices = (mask > 0)
    if not np.any(indices):
        channels = image.shape[2] if image.ndim == 3 else 1
        return np.empty((0, channels), dtype=image.dtype)

    skin_pixels = image[indices]
    return skin_pixels


# ==============================================================================
# 4. ĐÁNH GIÁ CHẤT LƯỢNG MẶT NẠ (MASK QUALITY EVALUATION)
# ==============================================================================

def evaluate_mask_quality(
    mask: np.ndarray,
    min_skin_ratio: float = MIN_SKIN_RATIO_THRESHOLD
) -> Dict[str, Any]:
    """
    Đánh giá định lượng chất lượng của mặt nạ phân đoạn da.
    Evaluate skin segmentation mask quality (percentage, connected components, validity).

    Args:
        mask: Mặt nạ nhị phân [H, W].
        min_skin_ratio: Tỷ lệ diện tích da tối thiểu để coi là hợp lệ (mặc định 5%).

    Returns:
        Dict[str, Any]: Báo cáo các chỉ số chất lượng:
            - total_pixels: Tổng số điểm ảnh.
            - skin_pixels: Số điểm ảnh da phát hiện được.
            - skin_percentage: Phần trăm diện tích da (%).
            - num_components: Số vùng liên thông riêng biệt.
            - largest_component_area: Diện tích của mảng da lớn nhất.
            - largest_component_ratio: Tỷ lệ diện tích mảng lớn nhất so với toàn bộ vùng da.
            - is_valid: True nếu mặt nạ đạt chuẩn chất lượng cho bước tiếp theo.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("[Lỗi] Mặt nạ phải là mảng 2 chiều [H, W].")

    total_pixels = int(mask.size)
    skin_pixels = int(np.count_nonzero(mask > 0))
    skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0.0

    # Phân tích thành phần liên thông
    bin_mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    if num_labels > 1:
        # Nhãn 0 là background, nhãn 1 trở đi là các mảng da
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = int(np.max(areas))
        largest_ratio = (largest_area / skin_pixels) if skin_pixels > 0 else 0.0
        num_components = num_labels - 1
    else:
        largest_area = 0
        largest_ratio = 0.0
        num_components = 0

    is_valid = bool((skin_ratio >= min_skin_ratio) and (num_components > 0))

    report = {
        "total_pixels": total_pixels,
        "skin_pixels": skin_pixels,
        "skin_ratio": round(skin_ratio, 4),
        "skin_percentage": round(skin_ratio * 100, 2),
        "num_components": num_components,
        "largest_component_area": largest_area,
        "largest_component_ratio": round(largest_ratio, 4),
        "is_valid": is_valid,
    }
    return report


# ==============================================================================
# 5. PIPELINES NHẬN DIỆN VÙNG DA (BASELINE & ENHANCED PIPELINES)
# ==============================================================================

def baseline_skin_detection(
    image: np.ndarray,
    color_space: str = "hsv",
    is_bgr: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Pipeline phát hiện da cơ bản (Baseline Skin Detection Pipeline):
    Chỉ dùng ngưỡng màu đơn giản (HSV hoặc YCrCb), không áp dụng hình thái học.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        color_space: 'hsv' hoặc 'ycrcb'.
        is_bgr: True nếu là định dạng BGR.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            - Mặt nạ nhị phân thô (mask).
            - Ảnh vùng da đã trích xuất (extracted skin).
            - Siêu dữ liệu và đánh giá chất lượng mặt nạ.
    """
    color_space = color_space.lower()
    if color_space == "hsv":
        mask, info = detect_skin_hsv(image, is_bgr=is_bgr)
    elif color_space in ("ycrcb", "ycbcr"):
        mask, info = detect_skin_ycrcb(image, is_bgr=is_bgr)
    else:
        raise ValueError(f"[Lỗi] Không gian màu không hỗ trợ cho Baseline: {color_space}")

    extracted, roi_meta = extract_roi(image, mask, crop_to_bbox=False)
    quality = evaluate_mask_quality(mask)

    metadata = {
        "pipeline": "baseline_skin_detection",
        "color_space": color_space,
        "color_info": info,
        "roi_meta": roi_meta,
        "quality": quality
    }
    return mask, extracted, metadata


def enhanced_skin_detection(
    image: np.ndarray,
    method: str = "combined",
    min_region_area: int = MIN_SKIN_REGION_AREA,
    fill_holes: bool = True,
    crop_to_bbox: bool = False,
    is_bgr: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Pipeline phát hiện da nâng cao (Enhanced Skin Detection Pipeline):
    Quy trình chuẩn:
    1. Ngưỡng kết hợp HSV và YCrCb (tối ưu hóa độ đặc hiệu và độ nhạy).
    2. Lọc hình thái học Opening để xóa nhiễu đốm nhỏ.
    3. Lọc hình thái học Closing để hàn gắn các mảng da đứt đoạn.
    4. Lấp đầy lỗ hổng bên trong (Hole Filling).
    5. Lọc diện tích thành phần liên thông (> min_region_area).
    6. Trích xuất ROI và đánh giá chất lượng mặt nạ.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        method: 'combined' (khuyến nghị), 'hsv', hoặc 'ycrcb'.
        min_region_area: Ngưỡng diện tích loại bỏ mảng nhiễu nhỏ.
        fill_holes: Lấp các lỗ rỗng bên trong vùng da mặt.
        crop_to_bbox: Cắt gọn khung ảnh theo hình chữ nhật bao quanh.
        is_bgr: Định dạng ảnh là BGR hay RGB.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            - Mặt nạ nhị phân tinh chỉnh (cleaned_mask).
            - Ảnh vùng da đã trích xuất (extracted_skin).
            - Siêu dữ liệu chi tiết kèm các mặt nạ trung gian.
    """
    # 1. Tạo mặt nạ thô theo ngưỡng kết hợp
    raw_mask, mask_meta = create_skin_mask(image, method=method, is_bgr=is_bgr)

    # 2. Làm sạch mặt nạ với hình thái học
    cleaned_mask, clean_meta = clean_skin_mask(
        raw_mask,
        min_region_area=min_region_area,
        fill_holes=fill_holes
    )

    # 3. Trích xuất ROI
    extracted, roi_meta = extract_roi(image, cleaned_mask, crop_to_bbox=crop_to_bbox)

    # 4. Đánh giá chất lượng
    quality = evaluate_mask_quality(cleaned_mask)

    metadata = {
        "pipeline": "enhanced_skin_detection",
        "method": method,
        "mask_creation_meta": mask_meta,
        "cleaning_meta": clean_meta,
        "roi_meta": roi_meta,
        "quality": quality,
        "raw_mask": raw_mask,
        "cleaned_mask": cleaned_mask
    }
    return cleaned_mask, extracted, metadata


# ==============================================================================
# 6. TRỰC QUAN HÓA KẾT QUẢ (VISUALIZATION)
# ==============================================================================

def visualize_skin_detection(
    image: np.ndarray,
    mask: np.ndarray,
    extracted: np.ndarray,
    title: str = "Kết Quả Phát Hiện Vùng Da (Skin Detection)",
    is_bgr: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Trực quan hóa so sánh 3 bảng điều khiển:
    [Ảnh gốc] -> [Mặt nạ nhị phân] -> [Vùng da đã trích xuất].
    Visualize 3-panel comparison: Original Image, Binary Skin Mask, Extracted Skin ROI.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        mask: Mặt nạ nhị phân [H, W].
        extracted: Ảnh vùng da đã trích xuất [H, W, 3].
        title: Tiêu đề chung cho biểu đồ.
        is_bgr: True nếu các ảnh màu đang ở định dạng BGR.
        save_path: Đường dẫn lưu file ảnh (tùy chọn).

    Returns:
        plt.Figure: Đối tượng đồ họa của Matplotlib.
    """
    # Chuyển đổi sang RGB để matplotlib hiển thị màu sắc chuẩn xác
    disp_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image.copy()
    disp_ext = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB) if is_bgr else extracted.copy()

    # Tính toán thông số đánh giá nhanh
    quality = evaluate_mask_quality(mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title}\n(Tỉ lệ da: {quality['skin_percentage']}% | Số vùng: {quality['num_components']})",
                 fontsize=14, fontweight="bold")

    # Bảng 1: Ảnh gốc
    axes[0].imshow(disp_img)
    axes[0].set_title("1. Ảnh Gốc (Original)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Bảng 2: Mặt nạ nhị phân
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("2. Mặt Nạ Nhị Phân (Binary Mask)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Bảng 3: Vùng da trích xuất
    axes[2].imshow(disp_ext)
    axes[2].set_title("3. Vùng Da Tách Biệt (Skin ROI)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200, bbox_inches="tight")

    return fig


__all__ = [
    "detect_skin_hsv",
    "detect_skin_ycrcb",
    "create_skin_mask",
    "apply_morphology",
    "fill_mask_holes",
    "clean_skin_mask",
    "extract_roi",
    "get_skin_pixels",
    "evaluate_mask_quality",
    "baseline_skin_detection",
    "enhanced_skin_detection",
    "visualize_skin_detection",
]

