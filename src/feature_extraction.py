"""
Module Trích Xuất Đặc Trưng Ảnh Da Mặt (Facial Skin Feature Extraction Module)
Dự án: Phân loại loại da mặt (Xử lý ảnh số - Digital Image Processing)

File: src/feature_extraction.py
Mục đích:
- Trích xuất 3 nhóm đặc trưng quan trọng cho phân loại loại da mặt:
    1. Nhóm A - Đặc trưng màu sắc (Color Features):
       - Giá trị trung bình (Mean) và độ lệch chuẩn (Std) từng kênh màu trên vùng da.
       - Lược đồ màu phân bổ (Color Histogram) chuẩn hóa với số bin cố định (32 bins).
       - Màu sắc chủ đạo (Dominant Color Extraction) trên vùng da.
    2. Nhóm B - Đặc trưng độ sáng (Brightness Features):
       - Độ sáng trung bình, độ lệch chuẩn độ sáng.
       - Các phân vị độ sáng (Percentiles: 25th, 50th / Median, 75th) và dải độ sáng (Range).
       - Tỷ lệ vùng chói sáng (Specular Highlight Ratio) đặc trưng cho da dầu.
    3. Nhóm C - Đặc trưng kết cấu bề mặt (Texture Features):
       - Phân bố hoa văn nhị phân cục bộ (Local Binary Pattern - LBP).
       - Lược đồ mẫu LBP chuẩn hóa (LBP Histogram).
       - Các chỉ số thống kê LBP (Mean, Std, Energy, Entropy).
- Kết hợp linh hoạt các nhóm đặc trưng thành vector đặc trưng duy nhất (Combined Feature Vector).
- Cung cấp danh sách tên đặc trưng (Feature Names) rõ ràng cho từng chiều trong vector.
- Xử lý triệt để các trường hợp biên (mặt nạ rỗng, không đủ điểm ảnh da).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import MiniBatchKMeans

# Nhập cấu hình và các module liên quan
try:
    from src import config
    from src.color_space import (
        convert_to_gray,
        convert_to_hsv,
        convert_to_ycrcb,
        split_channels,
    )
    from src.skin_detection import get_skin_pixels
except ImportError:
    # Cấu hình dự phòng an toàn
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.color_space import (
        convert_to_gray,
        convert_to_hsv,
        convert_to_ycrcb,
        split_channels,
    )
    from src.skin_detection import get_skin_pixels

logger = logging.getLogger("xla_pipeline.feature_extraction")

# Ngưỡng số lượng điểm ảnh da tối thiểu để trích xuất đặc trưng hợp lệ
MIN_VALID_SKIN_PIXELS: int = 50


# ==============================================================================
# 1. NHÓM A: ĐẶC TRƯNG MÀU SẮC TRÊN VÙNG DA (COLOR FEATURES - GROUP A)
# ==============================================================================

def extract_color_moments(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    color_space: str = "RGB",
    is_bgr: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Trích xuất giá trị Mean và Standard Deviation trên từng kênh màu, CHỈ tính trên điểm ảnh da.
    Extract color mean and std per channel on skin pixels only.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        mask: Mặt nạ nhị phân [H, W] (giá trị > 0 là da).
        color_space: 'RGB', 'HSV', hoặc 'YCbCr' / 'YCrCb'.
        is_bgr: True nếu định dạng ảnh là BGR, False nếu là RGB.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - Vector 1D chứa 6 đặc trưng: [mean_c1, mean_c2, mean_c3, std_c1, std_c2, std_c3].
            - Danh sách 6 tên đặc trưng tương ứng.
    """
    cs = color_space.upper()
    if cs in ("YCBCR", "YCRCB"):
        channel_names = ["Y", "Cr", "Cb"]
    elif cs == "HSV":
        channel_names = ["H", "S", "V"]
    else:
        cs = "RGB"
        channel_names = ["R", "G", "B"]

    feature_names = [f"color_{cs}_{ch}_mean" for ch in channel_names] + \
                    [f"color_{cs}_{ch}_std" for ch in channel_names]

    # Chuyển đổi không gian màu phù hợp
    if cs == "HSV":
        img_converted = convert_to_hsv(image, is_bgr=is_bgr)
    elif cs in ("YCBCR", "YCRCB"):
        img_converted = convert_to_ycrcb(image, is_bgr=is_bgr, standard_order=False)
    else:
        img_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image.copy()

    # Lấy tập hợp điểm ảnh da
    if mask is not None and np.any(mask > 0):
        skin_pixels = get_skin_pixels(img_converted, mask)
    else:
        skin_pixels = img_converted.reshape(-1, 3)

    if skin_pixels.shape[0] < MIN_VALID_SKIN_PIXELS:
        warnings.warn(
            f"[Cảnh báo] Quá ít điểm ảnh da ({skin_pixels.shape[0]} < {MIN_VALID_SKIN_PIXELS}) "
            "khi tính color moments. Trả về vector 0.",
            RuntimeWarning
        )
        return np.zeros(6, dtype=np.float32), feature_names

    means = np.mean(skin_pixels, axis=0, dtype=np.float64)
    stds = np.std(skin_pixels, axis=0, dtype=np.float64)

    vector = np.concatenate([means, stds]).astype(np.float32)
    return vector, feature_names


def extract_color_histogram(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    color_space: str = "RGB",
    bins: int = 32,
    hist_range: Tuple[int, int] = (0, 256),
    is_bgr: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Trích xuất biểu đồ phân bố màu (Color Histogram) chuẩn hóa cho từng kênh trên vùng da.
    Extract normalized color histogram per channel for skin pixels.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        mask: Mặt nạ nhị phân [H, W] lọc vùng da.
        color_space: 'RGB', 'HSV', hoặc 'YCbCr'.
        bins: Số lượng bin trong histogram (mặc định 32 bins theo config.HIST_BINS).
        hist_range: Miền giá trị điểm ảnh (0, 256).
        is_bgr: Định dạng ảnh là BGR hay RGB.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - Vector 1D có kích thước (3 * bins,) đã chuẩn hóa (tổng mỗi kênh = 1.0).
            - Danh sách tên đặc trưng tương ứng.
    """
    cs = color_space.upper()
    if cs in ("YCBCR", "YCRCB"):
        ch_names = ["Y", "Cr", "Cb"]
        img_converted = convert_to_ycrcb(image, is_bgr=is_bgr, standard_order=False)
    elif cs == "HSV":
        ch_names = ["H", "S", "V"]
        img_converted = convert_to_hsv(image, is_bgr=is_bgr)
    else:
        cs = "RGB"
        ch_names = ["R", "G", "B"]
        img_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image.copy()

    feature_names = [f"color_{cs}_{ch}_hist_bin_{b}" for ch in ch_names for b in range(bins)]
    total_features = 3 * bins

    if mask is not None and np.any(mask > 0):
        skin_pixels = get_skin_pixels(img_converted, mask)
    else:
        skin_pixels = img_converted.reshape(-1, 3)

    if skin_pixels.shape[0] < MIN_VALID_SKIN_PIXELS:
        warnings.warn(
            f"[Cảnh báo] Quá ít điểm ảnh da ({skin_pixels.shape[0]} < {MIN_VALID_SKIN_PIXELS}) "
            "khi tính color histogram. Trả về vector 0.",
            RuntimeWarning
        )
        return np.zeros(total_features, dtype=np.float32), feature_names

    histograms = []
    for c_idx in range(3):
        c_pixels = skin_pixels[:, c_idx]
        # Đối với kênh Hue trong HSV (0-179 trong OpenCV), điều chỉnh range phù hợp nếu cần
        cur_range = (0, 180) if (cs == "HSV" and c_idx == 0) else hist_range
        hist, _ = np.histogram(c_pixels, bins=bins, range=cur_range)
        # Chuẩn hóa histogram dạng phân phối xác suất L1-norm
        hist_sum = np.sum(hist)
        hist_norm = hist.astype(np.float32) / (hist_sum + 1e-7)
        histograms.append(hist_norm)

    vector = np.concatenate(histograms).astype(np.float32)
    return vector, feature_names


def extract_dominant_color(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    color_space: str = "RGB",
    k: int = 1,
    max_pixels: int = 5000,
    is_bgr: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Trích xuất màu sắc chủ đạo (Dominant Color) của vùng da thông qua thuật toán phân cụm K-Means.
    Extract dominant color(s) on skin pixels using MiniBatchKMeans.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        mask: Mặt nạ phân đoạn da [H, W].
        color_space: 'RGB', 'HSV', hoặc 'YCbCr'.
        k: Số màu chủ đạo cần trích xuất (mặc định k=1 trả về 3 tọa độ màu của cụm lớn nhất).
        max_pixels: Giới hạn số pixel ngẫu nhiên đưa vào K-Means để tối ưu tốc độ tính toán.
        is_bgr: True nếu ảnh đầu vào là BGR.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - Vector 1D chứa (3 * k) giá trị màu chủ đạo.
            - Danh sách tên đặc trưng tương ứng.
    """
    cs = color_space.upper()
    if cs in ("YCBCR", "YCRCB"):
        ch_names = ["Y", "Cr", "Cb"]
        img_converted = convert_to_ycrcb(image, is_bgr=is_bgr, standard_order=False)
    elif cs == "HSV":
        ch_names = ["H", "S", "V"]
        img_converted = convert_to_hsv(image, is_bgr=is_bgr)
    else:
        cs = "RGB"
        ch_names = ["R", "G", "B"]
        img_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image.copy()

    feature_names = []
    for cluster_i in range(k):
        for ch in ch_names:
            feature_names.append(f"color_{cs}_dom{cluster_i+1}_{ch}")

    if mask is not None and np.any(mask > 0):
        skin_pixels = get_skin_pixels(img_converted, mask)
    else:
        skin_pixels = img_converted.reshape(-1, 3)

    if skin_pixels.shape[0] < max(MIN_VALID_SKIN_PIXELS, k):
        warnings.warn(
            f"[Cảnh báo] Quá ít điểm ảnh da ({skin_pixels.shape[0]}) cho dominant color. "
            "Trả về vector 0.",
            RuntimeWarning
        )
        return np.zeros(3 * k, dtype=np.float32), feature_names

    # Lấy mẫu ngẫu nhiên nếu số pixel quá lớn để tăng tốc độ phân cụm
    if skin_pixels.shape[0] > max_pixels:
        rng = np.random.RandomState(config.RANDOM_STATE if hasattr(config, "RANDOM_STATE") else 42)
        idx_sample = rng.choice(skin_pixels.shape[0], size=max_pixels, replace=False)
        sample_pixels = skin_pixels[idx_sample].astype(np.float32)
    else:
        sample_pixels = skin_pixels.astype(np.float32)

    try:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=config.RANDOM_STATE if hasattr(config, "RANDOM_STATE") else 42,
            batch_size=min(512, len(sample_pixels)),
            n_init=3
        )
        kmeans.fit(sample_pixels)
        centers = kmeans.cluster_centers_  # [k, 3]

        # Sắp xếp các cụm theo tần suất xuất hiện (cụm đông điểm ảnh nhất đứng đầu)
        counts = np.bincount(kmeans.labels_, minlength=k)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_centers = centers[sorted_indices]

        vector = sorted_centers.flatten().astype(np.float32)
    except Exception as exc:
        logger.warning(f"Lỗi khi chạy MiniBatchKMeans: {exc}. Dùng trung bình pixel thay thế.")
        mean_val = np.mean(sample_pixels, axis=0)
        vector = np.tile(mean_val, k).astype(np.float32)

    return vector, feature_names


def extract_group_a_color_features(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    color_space: str = "RGB",
    bins: int = 32,
    k_dominant: int = 1,
    is_bgr: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Trích xuất TOÀN BỘ đặc trưng màu sắc Nhóm A:
    Moments (6) + Histogram (3 * bins) + Dominant Color (3 * k).
    Extract complete Group A color feature vector.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        mask: Mặt nạ phân đoạn da [H, W].
        color_space: 'RGB', 'HSV', hoặc 'YCbCr'.
        bins: Số bins của histogram (mặc định 32).
        k_dominant: Số màu chủ đạo (mặc định 1).
        is_bgr: Định dạng ảnh là BGR hay RGB.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - Vector 1D đặc trưng nhóm A (tổng 6 + 96 + 3 = 105 chiều với bins=32, k=1).
            - Danh sách tên đặc trưng đầy đủ.
    """
    v_moments, names_moments = extract_color_moments(
        image, mask=mask, color_space=color_space, is_bgr=is_bgr
    )
    v_hist, names_hist = extract_color_histogram(
        image, mask=mask, color_space=color_space, bins=bins, is_bgr=is_bgr
    )
    v_dom, names_dom = extract_dominant_color(
        image, mask=mask, color_space=color_space, k=k_dominant, is_bgr=is_bgr
    )

    vector = np.concatenate([v_moments, v_hist, v_dom]).astype(np.float32)
    names = names_moments + names_hist + names_dom
    return vector, names


# ==============================================================================
# 2. NHÓM B: ĐẶC TRƯNG ĐỘ SÁNG TRÊN VÙNG DA (BRIGHTNESS FEATURES - GROUP B)
# ==============================================================================

def extract_brightness_features(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    source: str = "HSV_V",
    highlight_thresh: int = 230,
    is_bgr: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Trích xuất các đặc trưng định lượng độ sáng trên vùng da (Nhóm B).
    Extract brightness descriptors on skin pixels:
    - Độ sáng trung bình (Mean Brightness).
    - Độ lệch chuẩn độ sáng (Brightness Std).
    - Các phân vị độ sáng (Percentiles 25th, 50th / Median, 75th).
    - Dải độ sáng (Brightness Range = max - min).
    - Tỷ lệ vùng phản chiếu chói sáng dầu nhờn (Highlight Ratio: V > highlight_thresh).

    Đặc trưng độ sáng và độ chói phản xạ là tín hiệu thị giác quan trọng nhất
    để phân biệt da dầu (nhiều vùng bóng nhờn, độ sáng phân tán rộng) và
    da khô (độ sáng đồng đều, mờ lì, ít phản chiếu).

    Args:
        image: Ảnh đầu vào [H, W, 3].
        mask: Mặt nạ lọc vùng da [H, W].
        source: Kênh lấy độ sáng ('HSV_V', 'YCBCR_Y', hoặc 'GRAY').
        highlight_thresh: Ngưỡng phát hiện chói sáng (mặc định 230 theo config).
        is_bgr: True nếu là BGR, False nếu là RGB.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - Vector 1D gồm 7 đặc trưng độ sáng.
            - Danh sách 7 tên đặc trưng tương ứng.
    """
    src_upper = source.upper()
    if "Y" in src_upper:
        ycrcb = convert_to_ycrcb(image, is_bgr=is_bgr)
        brightness_map = ycrcb[:, :, 0]  # Kênh Y (Luma)
        tag = "Y"
    elif "GRAY" in src_upper:
        brightness_map = convert_to_gray(image, is_bgr=is_bgr)
        tag = "Gray"
    else:
        hsv = convert_to_hsv(image, is_bgr=is_bgr)
        brightness_map = hsv[:, :, 2]  # Kênh V (Value)
        tag = "V"

    feature_names = [
        f"brightness_{tag}_mean",
        f"brightness_{tag}_std",
        f"brightness_{tag}_p25",
        f"brightness_{tag}_p50_median",
        f"brightness_{tag}_p75",
        f"brightness_{tag}_range",
        f"brightness_{tag}_highlight_ratio",
    ]

    if mask is not None and np.any(mask > 0):
        skin_values = brightness_map[mask > 0].astype(np.float64)
    else:
        skin_values = brightness_map.ravel().astype(np.float64)

    if skin_values.size < MIN_VALID_SKIN_PIXELS:
        warnings.warn(
            f"[Cảnh báo] Quá ít điểm ảnh da ({skin_values.size} < {MIN_VALID_SKIN_PIXELS}) "
            "khi tính brightness features. Trả về vector 0.",
            RuntimeWarning
        )
        return np.zeros(len(feature_names), dtype=np.float32), feature_names

    b_mean = float(np.mean(skin_values))
    b_std = float(np.std(skin_values))
    p25, p50, p75 = np.percentile(skin_values, [25.0, 50.0, 75.0])
    b_range = float(np.max(skin_values) - np.min(skin_values))
    highlight_ratio = float(np.count_nonzero(skin_values >= highlight_thresh) / skin_values.size)

    vector = np.array([
        b_mean,
        b_std,
        float(p25),
        float(p50),
        float(p75),
        b_range,
        highlight_ratio
    ], dtype=np.float32)

    return vector, feature_names


# ==============================================================================
# 3. NHÓM C: ĐẶC TRƯNG KẾT CẤU BỀ MẶT DA (TEXTURE LBP - GROUP C)
# ==============================================================================

def extract_lbp_features(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    radius: int = 1,
    n_points: int = 8,
    method: str = "uniform",
    is_bgr: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Trích xuất đặc trưng kết cấu bề mặt da bằng toán tử LBP (Local Binary Pattern).
    Extract skin surface texture features using Local Binary Pattern:
    - LBP Histogram (chuẩn hóa trên vùng da, với phương pháp 'uniform' có n_points + 2 bins).
    - Thống kê LBP (Mean, Std, Energy / Đồng nhất, Entropy / Độ phức tạp).

    LBP phản ánh cấu trúc vi mô biểu bì:
    - Da khô thường có các vảy thô ráp, hoa văn không đồng đều, độ Entropy LBP cao.
    - Da thường có vi cấu trúc đồng nhất, mịn màng, độ Energy LBP cao hơn.

    Args:
        image: Ảnh đầu vào [H, W, 3] hoặc [H, W].
        mask: Mặt nạ phân đoạn da [H, W].
        radius: Bán kính lân cận R (mặc định 1 theo config.LBP_RADIUS).
        n_points: Số điểm lân cận P (mặc định 8 theo config.LBP_N_POINTS).
        method: Phương pháp LBP ('uniform', 'default', 'ror'). Mặc định 'uniform'.
        is_bgr: True nếu ảnh đầu vào là BGR.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - Vector 1D gồm LBP Histogram (P + 2 chiều) + LBP Statistics (4 chiều).
            - Danh sách tên đặc trưng đầy đủ.
    """
    # 1. Chuyển đổi sang ảnh mức xám
    gray = convert_to_gray(image, is_bgr=is_bgr)

    # 2. Tính toán ma trận LBP trên toàn ảnh
    lbp_map = local_binary_pattern(gray, P=n_points, R=radius, method=method)

    # Xác định số bin của LBP histogram
    if method == "uniform":
        n_bins = n_points + 2
    else:
        n_bins = 2 ** n_points

    hist_names = [f"texture_lbp_hist_bin_{b}" for b in range(n_bins)]
    stat_names = [
        "texture_lbp_mean",
        "texture_lbp_std",
        "texture_lbp_energy",
        "texture_lbp_entropy"
    ]
    feature_names = hist_names + stat_names
    total_dim = n_bins + 4

    # 3. Lấy giá trị LBP trên các điểm ảnh thuộc vùng da
    if mask is not None and np.any(mask > 0):
        lbp_skin = lbp_map[mask > 0].ravel()
    else:
        lbp_skin = lbp_map.ravel()

    if lbp_skin.size < MIN_VALID_SKIN_PIXELS:
        warnings.warn(
            f"[Cảnh báo] Quá ít điểm ảnh da ({lbp_skin.size} < {MIN_VALID_SKIN_PIXELS}) "
            "khi tính LBP texture. Trả về vector 0.",
            RuntimeWarning
        )
        return np.zeros(total_dim, dtype=np.float32), feature_names

    # 4. Tính histogram LBP chuẩn hóa
    hist, _ = np.histogram(lbp_skin, bins=n_bins, range=(0, n_bins))
    hist_sum = np.sum(hist)
    hist_norm = hist.astype(np.float32) / (hist_sum + 1e-7)

    # 5. Tính các thống kê kết cấu
    lbp_mean = float(np.mean(lbp_skin))
    lbp_std = float(np.std(lbp_skin))
    # Năng lượng (Energy / Uniformity): Tổng bình phương xác suất các bin
    lbp_energy = float(np.sum(hist_norm ** 2))
    # Entropy (Độ bất định / Độ thô ráp bề mặt)
    hist_prob = hist_norm[hist_norm > 0]
    lbp_entropy = float(-np.sum(hist_prob * np.log2(hist_prob + 1e-10)))

    stats_vector = np.array([lbp_mean, lbp_std, lbp_energy, lbp_entropy], dtype=np.float32)
    vector = np.concatenate([hist_norm, stats_vector]).astype(np.float32)

    return vector, feature_names


# ==============================================================================
# 4. TỔNG HỢP VECTOR ĐẶC TRƯNG VÀ TÊN CHIỀU (COMBINED FEATURE EXTRACTION)
# ==============================================================================

def get_feature_names(
    feature_group: str = "all",
    color_space: str = "RGB",
    hist_bins: int = 32,
    k_dominant: int = 1,
    n_points: int = 8,
    lbp_method: str = "uniform"
) -> List[str]:
    """
    Trả về danh sách đầy đủ tên các đặc trưng tương ứng với cấu hình trích xuất.
    Get ordered list of feature names corresponding to feature extraction settings.

    Args:
        feature_group: Cấu hình nhóm đặc trưng:
            - 'color': Chỉ nhóm A
            - 'brightness': Chỉ nhóm B
            - 'texture': Chỉ nhóm C
            - 'color_brightness': Nhóm A + B
            - 'color_texture': Nhóm A + C
            - 'brightness_texture': Nhóm B + C
            - 'all' hoặc 'color_brightness_texture': Toàn bộ nhóm A + B + C
        color_space: 'RGB', 'HSV', hoặc 'YCbCr'.
        hist_bins: Số bin histogram màu.
        k_dominant: Số màu chủ đạo.
        n_points: Số điểm lân cận LBP.
        lbp_method: Phương pháp LBP.

    Returns:
        List[str]: Danh sách tên từng vị trí trong vector đặc trưng.
    """
    fg = feature_group.lower()
    names: List[str] = []

    # Nhóm A: Color
    if fg in ("color", "color_brightness", "color_texture", "all", "color_brightness_texture"):
        cs = color_space.upper()
        ch_names = ["Y", "Cr", "Cb"] if cs in ("YCBCR", "YCRCB") else (["H", "S", "V"] if cs == "HSV" else ["R", "G", "B"])
        # Moments
        names.extend([f"color_{cs}_{ch}_mean" for ch in ch_names])
        names.extend([f"color_{cs}_{ch}_std" for ch in ch_names])
        # Hist
        names.extend([f"color_{cs}_{ch}_hist_bin_{b}" for ch in ch_names for b in range(hist_bins)])
        # Dominant
        for i in range(k_dominant):
            for ch in ch_names:
                names.append(f"color_{cs}_dom{i+1}_{ch}")

    # Nhóm B: Brightness
    if fg in ("brightness", "color_brightness", "brightness_texture", "all", "color_brightness_texture"):
        tag = "V" if color_space.upper() == "HSV" else ("Y" if color_space.upper() in ("YCBCR", "YCRCB") else "V")
        names.extend([
            f"brightness_{tag}_mean",
            f"brightness_{tag}_std",
            f"brightness_{tag}_p25",
            f"brightness_{tag}_p50_median",
            f"brightness_{tag}_p75",
            f"brightness_{tag}_range",
            f"brightness_{tag}_highlight_ratio",
        ])

    # Nhóm C: Texture
    if fg in ("texture", "color_texture", "brightness_texture", "all", "color_brightness_texture"):
        n_bins = (n_points + 2) if lbp_method == "uniform" else (2 ** n_points)
        names.extend([f"texture_lbp_hist_bin_{b}" for b in range(n_bins)])
        names.extend([
            "texture_lbp_mean",
            "texture_lbp_std",
            "texture_lbp_energy",
            "texture_lbp_entropy"
        ])

    return names


def extract_features(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    feature_group: str = "all",
    color_space: str = "RGB",
    hist_bins: int = 32,
    k_dominant: int = 1,
    lbp_radius: int = 1,
    lbp_n_points: int = 8,
    lbp_method: str = "uniform",
    is_bgr: bool = True
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Hàm giao diện chính trích xuất vector đặc trưng hoàn chỉnh từ ảnh và mặt nạ da.
    Main API for feature extraction with configurable feature groups.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        mask: Mặt nạ phân đoạn da [H, W]. Nếu None, áp dụng trên toàn bộ ảnh.
        feature_group: Nhóm đặc trưng trích xuất ('color', 'brightness', 'texture',
                       'color_brightness', 'all' / 'color_brightness_texture').
        color_space: Không gian màu ('RGB', 'HSV', 'YCbCr').
        hist_bins: Số bin color histogram (mặc định 32).
        k_dominant: Số màu chủ đạo (mặc định 1).
        lbp_radius: Bán kính LBP (mặc định 1).
        lbp_n_points: Số điểm lân cận LBP (mặc định 8).
        lbp_method: Phương pháp LBP ('uniform').
        is_bgr: True nếu ảnh đầu vào là BGR.

    Returns:
        Tuple[np.ndarray, List[str], Dict[str, Any]]:
            - features: Vector đặc trưng 1D dạng np.float32.
            - feature_names: Danh sách tên đặc trưng tương ứng.
            - info: Từ điển chi tiết chứa số pixel da, tỷ lệ diện tích, và từng nhóm con.
    """
    fg = feature_group.lower()
    expected_names = get_feature_names(
        feature_group=fg,
        color_space=color_space,
        hist_bins=hist_bins,
        k_dominant=k_dominant,
        n_points=lbp_n_points,
        lbp_method=lbp_method
    )

    # Đếm số pixel da thực tế
    skin_pixels_count = int(np.count_nonzero(mask > 0)) if mask is not None else int(image.shape[0] * image.shape[1])
    is_valid_mask = skin_pixels_count >= MIN_VALID_SKIN_PIXELS

    feature_subvectors: List[np.ndarray] = []
    meta_subgroups: Dict[str, Any] = {}

    # 1. Trích xuất nhóm màu sắc (Group A)
    if fg in ("color", "color_brightness", "color_texture", "all", "color_brightness_texture"):
        v_color, _ = extract_group_a_color_features(
            image, mask=mask, color_space=color_space, bins=hist_bins,
            k_dominant=k_dominant, is_bgr=is_bgr
        )
        feature_subvectors.append(v_color)
        meta_subgroups["color_dim"] = len(v_color)

    # 2. Trích xuất nhóm độ sáng (Group B)
    if fg in ("brightness", "color_brightness", "brightness_texture", "all", "color_brightness_texture"):
        source_channel = "HSV_V" if color_space.upper() == "HSV" else ("YCBCR_Y" if color_space.upper() in ("YCBCR", "YCRCB") else "HSV_V")
        v_bright, _ = extract_brightness_features(
            image, mask=mask, source=source_channel, is_bgr=is_bgr
        )
        feature_subvectors.append(v_bright)
        meta_subgroups["brightness_dim"] = len(v_bright)

    # 3. Trích xuất nhóm kết cấu (Group C)
    if fg in ("texture", "color_texture", "brightness_texture", "all", "color_brightness_texture"):
        v_tex, _ = extract_lbp_features(
            image, mask=mask, radius=lbp_radius, n_points=lbp_n_points,
            method=lbp_method, is_bgr=is_bgr
        )
        feature_subvectors.append(v_tex)
        meta_subgroups["texture_dim"] = len(v_tex)

    if not feature_subvectors:
        raise ValueError(f"[Lỗi] Nhóm đặc trưng '{feature_group}' không được hỗ trợ.")

    combined_vector = np.concatenate(feature_subvectors).astype(np.float32)

    # Đảm bảo số lượng chiều vector khớp chính xác với feature_names
    if combined_vector.shape[0] != len(expected_names):
        logger.warning(
            f"Kích thước vector ({combined_vector.shape[0]}) khác số tên ({len(expected_names)}). "
            "Đang đồng bộ hóa tên đặc trưng."
        )
        expected_names = [f"feature_{i}" for i in range(combined_vector.shape[0])]

    info = {
        "feature_group": fg,
        "color_space": color_space,
        "skin_pixels_count": skin_pixels_count,
        "is_valid_mask": is_valid_mask,
        "feature_dim": int(combined_vector.shape[0]),
        "subgroups": meta_subgroups,
    }

    return combined_vector, expected_names, info
