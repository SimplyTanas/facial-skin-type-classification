"""
Module Chuyển Đổi và Phân Tích Không Gian Màu (Color Space Conversion & Analysis Module)
Dự án: Phân loại loại da mặt (Xử lý ảnh số - Digital Image Processing)

File: src/color_space.py
Mục đích:
- Chuyển đổi linh hoạt giữa các không gian màu: RGB, HSV, YCrCb (YCbCr), Grayscale, CIELAB.
- Tách kênh màu độc lập và trực quan hóa từng thành phần sắc độ / độ sáng.
- So sánh song song các không gian màu để phục vụ bài giảng và báo cáo đồ án.
- Tính toán các chỉ số thống kê màu sắc (Mean, Std, Median, Min, Max, Skewness).
- Trực quan hóa phân bố màu (1D & 2D Histograms) của vùng da mặt.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Nhập cấu hình từ src.config nếu có
try:
    from src.config import TARGET_SIZE
except ImportError:
    TARGET_SIZE = (224, 224)


# ==============================================================================
# 1. CÁC HÀM CHUYỂN ĐỔI KHÔNG GIAN MÀU (COLOR CONVERSIONS)
# ==============================================================================

def convert_to_hsv(image: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    """
    Chuyển đổi ảnh sang không gian màu HSV (Hue - Saturation - Value).
    Convert image to HSV color space.
    - H (Sắc tướng: 0-179): Độc lập với độ chiếu sáng, rất nhạy với sắc tố da.
    - S (Độ bão hòa: 0-255): Đo độ đậm nhạt của màu sắc.
    - V (Độ sáng: 0-255): Thể hiện cường độ ánh sáng, có thể phản ánh vùng da dầu bóng nhờn.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        is_bgr: True nếu ảnh đầu vào là BGR, False nếu là RGB.

    Returns:
        np.ndarray: Ảnh trong không gian màu HSV [H, W, 3] (uint8).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("[Lỗi] Ảnh đầu vào cần có 3 kênh màu để chuyển sang HSV.")
    cvt_code = cv2.COLOR_BGR2HSV if is_bgr else cv2.COLOR_RGB2HSV
    return cv2.cvtColor(image, cvt_code)


def convert_to_ycrcb(
    image: np.ndarray,
    is_bgr: bool = True,
    standard_order: bool = False
) -> np.ndarray:
    """
    Chuyển đổi ảnh sang không gian màu YCrCb / YCbCr.
    Convert image to YCrCb or standard YCbCr color space.
    - Y (Luma): Kênh độ chói (độ sáng).
    - Cb: Hiệu màu lam (Chroma Blue).
    - Cr: Hiệu màu đỏ (Chroma Red).

    Đặc tính vượt trội: Tách biệt hoàn toàn thành phần độ sáng (Y) và màu sắc (Cr, Cb),
    giúp gom cụm màu da rất ổn định bất chấp sự biến đổi cường độ chiếu sáng.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        is_bgr: True nếu ảnh là BGR, False nếu là RGB.
        standard_order: Nếu True, đổi thứ tự kênh thành (Y, Cb, Cr) chuẩn quốc tế.
                        Nếu False (mặc định), giữ thứ tự chuẩn OpenCV là (Y, Cr, Cb).

    Returns:
        np.ndarray: Ảnh trong không gian màu YCrCb hoặc YCbCr [H, W, 3].
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("[Lỗi] Ảnh đầu vào cần có 3 kênh màu để chuyển sang YCrCb.")
    cvt_code = cv2.COLOR_BGR2YCrCb if is_bgr else cv2.COLOR_RGB2YCrCb
    ycrcb = cv2.cvtColor(image, cvt_code)

    if standard_order:
        # Đổi từ (Y, Cr, Cb) sang (Y, Cb, Cr)
        y = ycrcb[:, :, 0]
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        return np.stack([y, cb, cr], axis=2)

    return ycrcb


def convert_to_gray(image: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    """
    Chuyển đổi ảnh sang ảnh mức xám (Grayscale).
    Convert image to single-channel Grayscale.

    Args:
        image: Ảnh đầu vào [H, W, 3] hoặc [H, W].
        is_bgr: True nếu ảnh là BGR, False nếu là RGB.

    Returns:
        np.ndarray: Ảnh mức xám 1 kênh [H, W] (uint8).
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("[Lỗi] Đầu vào không phải mảng numpy hợp lệ.")
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 1:
        return image.squeeze()

    cvt_code = cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY
    return cv2.cvtColor(image, cvt_code)


def convert_to_lab(image: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    """
    Chuyển đổi ảnh sang không gian màu cảm nhận đều CIELAB (L*a*b*).
    Convert image to CIELAB color space.
    - L*: Độ sáng cảm nhận (Lightness: 0 - 100/255 trong OpenCV).
    - a*: Trục lục - đỏ (Green to Red).
    - b*: Trục lam - vàng (Blue to Yellow).
    Rất phổ biến trong da liễu học để tính góc ITA (Individual Typology Angle).

    Args:
        image: Ảnh đầu vào [H, W, 3].
        is_bgr: True nếu là BGR, False nếu là RGB.

    Returns:
        np.ndarray: Ảnh trong không gian màu CIELAB [H, W, 3].
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("[Lỗi] Ảnh đầu vào cần có 3 kênh màu để chuyển sang LAB.")
    cvt_code = cv2.COLOR_BGR2Lab if is_bgr else cv2.COLOR_RGB2Lab
    return cv2.cvtColor(image, cvt_code)


# ==============================================================================
# 2. TÁCH KÊNH VÀ TRỰC QUAN HÓA KÊNH MÀU (CHANNEL SPLITTING & VISUALIZATION)
# ==============================================================================

def split_channels(
    image: np.ndarray,
    color_space: str = "RGB",
    is_bgr: bool = True
) -> Dict[str, np.ndarray]:
    """
    Tách các kênh màu thành từng ma trận riêng biệt kèm tên kênh tương ứng.
    Split image into labeled individual channels based on chosen color space.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        color_space: 'RGB', 'HSV', 'YCRCB', 'YCBCR', hoặc 'LAB'.
        is_bgr: True nếu ảnh đầu vào là BGR, False nếu là RGB.

    Returns:
        Dict[str, np.ndarray]: Từ điển tên kênh -> mảng 2D mức xám [H, W].
    """
    cs = color_space.upper()

    if cs == "RGB":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image
        return {"R (Đỏ)": rgb[:, :, 0], "G (Lục)": rgb[:, :, 1], "B (Lam)": rgb[:, :, 2]}

    elif cs == "HSV":
        hsv = convert_to_hsv(image, is_bgr=is_bgr)
        return {"H (Sắc tướng)": hsv[:, :, 0], "S (Bão hòa)": hsv[:, :, 1], "V (Độ sáng)": hsv[:, :, 2]}

    elif cs in ("YCRCB", "YCBCR"):
        ycrcb = convert_to_ycrcb(image, is_bgr=is_bgr, standard_order=False)
        return {"Y (Độ chói)": ycrcb[:, :, 0], "Cr (Hiệu đỏ)": ycrcb[:, :, 1], "Cb (Hiệu lam)": ycrcb[:, :, 2]}

    elif cs == "LAB":
        lab = convert_to_lab(image, is_bgr=is_bgr)
        return {"L (Độ sáng)": lab[:, :, 0], "a (Lục-Đỏ)": lab[:, :, 1], "b (Lam-Vàng)": lab[:, :, 2]}

    else:
        raise ValueError(f"[Lỗi] Không gian màu không được hỗ trợ: {color_space}")


def visualize_channels(
    image: np.ndarray,
    color_space: str = "RGB",
    is_bgr: bool = True,
    cmap: str = "gray",
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Trực quan hóa ảnh gốc và từng kênh màu độc lập kèm biểu đồ sắc độ tương ứng.
    Visualize original image along with separate channels and their histograms.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        color_space: 'RGB', 'HSV', 'YCrCb', hoặc 'LAB'.
        is_bgr: Định dạng ảnh là BGR hay RGB.
        cmap: Bảng màu hiển thị kênh (mặc định 'gray').
        save_path: Đường dẫn lưu ảnh (tùy chọn).

    Returns:
        plt.Figure: Đối tượng đồ họa của Matplotlib.
    """
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image
    channels_dict = split_channels(image, color_space=color_space, is_bgr=is_bgr)
    channel_names = list(channels_dict.keys())

    # Tạo lưới 2 hàng x 4 cột:
    # Hàng 1: Ảnh gốc + 3 kênh ảnh xám
    # Hàng 2: Histogram ảnh gốc + 3 histogram từng kênh
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Phân Tích Chi Tiết Kênh Màu - Không Gian: {color_space.upper()}",
                 fontsize=14, fontweight="bold")

    # [0, 0]: Ảnh màu gốc
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title("Ảnh Màu Gốc (Original)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # [1, 0]: Histogram mức xám tổng thể
    gray = convert_to_gray(image, is_bgr=is_bgr)
    axes[1, 0].hist(gray.ravel(), bins=64, range=(0, 256), color="gray", alpha=0.7)
    axes[1, 0].set_title("Phân Bố Mức Xám (Gray)", fontsize=10)
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 3 kênh màu tiếp theo
    colors_hist = ["crimson", "forestgreen", "royalblue"]
    for idx, name in enumerate(channel_names):
        ch_img = channels_dict[name]

        # Hàng 1: Ảnh của kênh
        im = axes[0, idx + 1].imshow(ch_img, cmap=cmap)
        axes[0, idx + 1].set_title(f"Kênh {name}", fontsize=11, fontweight="bold")
        axes[0, idx + 1].axis("off")
        fig.colorbar(im, ax=axes[0, idx + 1], fraction=0.046, pad=0.04)

        # Hàng 2: Histogram của kênh
        c = colors_hist[idx % len(colors_hist)]
        axes[1, idx + 1].hist(ch_img.ravel(), bins=64, range=(0, 256), color=c, alpha=0.7)
        mean_val = np.mean(ch_img)
        std_val = np.std(ch_img)
        axes[1, idx + 1].set_title(f"μ={mean_val:.1f}, σ={std_val:.1f}", fontsize=10)
        axes[1, idx + 1].set_xlim([0, 256])
        axes[1, idx + 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200, bbox_inches="tight")

    return fig


# ==============================================================================
# 3. SO SÁNH CÁC KHÔNG GIAN MÀU SONG SONG (COLOR SPACES COMPARISON)
# ==============================================================================

def compare_color_spaces(
    image: np.ndarray,
    is_bgr: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    So sánh trực quan song song các không gian màu phổ biến trong xử lý ảnh da mặt:
    RGB, HSV, YCrCb, CIELAB và Grayscale.
    Side-by-side comparison of color space representations for image processing.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        is_bgr: True nếu ảnh đầu vào là BGR.
        save_path: Đường dẫn lưu ảnh biểu đồ (tùy chọn).

    Returns:
        plt.Figure: Đối tượng đồ họa Matplotlib.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image
    hsv = convert_to_hsv(image, is_bgr=is_bgr)
    ycrcb = convert_to_ycrcb(image, is_bgr=is_bgr, standard_order=False)
    lab = convert_to_lab(image, is_bgr=is_bgr)
    gray = convert_to_gray(image, is_bgr=is_bgr)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("So Sánh Các Không Gian Màu (Color Space Comparison)", fontsize=14, fontweight="bold")

    # 1. RGB
    axes[0].imshow(rgb)
    axes[0].set_title("1. RGB (Chuẩn hiển thị)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. HSV (Hiển thị như RGB giả lập để quan sát)
    axes[1].imshow(hsv)
    axes[1].set_title("2. HSV (Sắc-Bão hòa-Sáng)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. YCrCb
    axes[2].imshow(ycrcb)
    axes[2].set_title("3. YCrCb (Khử nhạy sáng)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4. CIELAB
    axes[3].imshow(lab)
    axes[3].set_title("4. LAB (Cảm nhận đều)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # 5. Grayscale
    axes[4].imshow(gray, cmap="gray")
    axes[4].set_title("5. Mức Xám (Grayscale)", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200, bbox_inches="tight")

    return fig


# ==============================================================================
# 4. TRÍCH XUẤT THỐNG KÊ MÀU SẮC (COLOR STATISTICS PER CHANNEL)
# ==============================================================================

def extract_color_statistics(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    color_space: str = "RGB",
    is_bgr: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Trích xuất các chỉ số thống kê phân bố màu sắc theo từng kênh màu.
    Extract statistical descriptors per channel: Mean, Std, Median, Min, Max, Skewness.

    Nếu cung cấp mặt nạ (mask), các thống kê chỉ được tính toán trên các điểm ảnh
    thuộc vùng da (Skin Region), giúp loại trừ nền và các yếu tố gây nhiễu.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        mask: Mặt nạ nhị phân [H, W] (tùy chọn, giá trị > 0 là vùng da).
        color_space: 'RGB', 'HSV', 'YCrCb', hoặc 'LAB'.
        is_bgr: True nếu ảnh đầu vào là BGR.

    Returns:
        Dict[str, Dict[str, float]]:
            Từ điển chứa tên kênh và các chỉ số thống kê tương ứng.
    """
    channels_dict = split_channels(image, color_space=color_space, is_bgr=is_bgr)
    results: Dict[str, Dict[str, float]] = {}

    for ch_name, ch_data in channels_dict.items():
        if mask is not None:
            if mask.shape[:2] != ch_data.shape[:2]:
                raise ValueError("[Lỗi] Kích thước mặt nạ không khớp kích thước ảnh.")
            valid_pixels = ch_data[mask > 0]
        else:
            valid_pixels = ch_data.ravel()

        if valid_pixels.size == 0:
            results[ch_name] = {
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "skewness": 0.0,
                "count": 0
            }
            continue

        valid_float = valid_pixels.astype(np.float64)
        mean_val = float(np.mean(valid_float))
        std_val = float(np.std(valid_float))
        median_val = float(np.median(valid_float))
        min_val = float(np.min(valid_float))
        max_val = float(np.max(valid_float))

        # Tính độ lệch (Skewness: mô men chuẩn hóa bậc 3)
        if std_val > 1e-6:
            skewness = float(np.mean(((valid_float - mean_val) / std_val) ** 3))
        else:
            skewness = 0.0

        results[ch_name] = {
            "mean": round(mean_val, 3),
            "std": round(std_val, 3),
            "median": round(median_val, 3),
            "min": round(min_val, 3),
            "max": round(max_val, 3),
            "skewness": round(skewness, 3),
            "count": int(valid_pixels.size)
        }

    return results


# ==============================================================================
# 5. TRỰC QUAN HÓA PHÂN BỐ MÀU VÙNG DA (COLOR DISTRIBUTION VISUALIZATION)
# ==============================================================================

def visualize_color_distribution(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    color_space: str = "HSV",
    bins: int = 40,
    is_bgr: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Trực quan hóa phân bố màu sắc của vùng da trong không gian màu:
    - 3 Biểu đồ Histogram 1D cho từng kênh.
    - 1 Biểu đồ phân bố liên hợp 2D (2D Joint Histogram / Scatter density)
      (ví dụ: H vs S trong HSV, hoặc Cb vs Cr trong YCbCr).
    Rất hữu ích cho báo cáo đồ án nhằm chứng minh đặc tính gom cụm màu của da mặt.

    Args:
        image: Ảnh đầu vào [H, W, 3].
        mask: Mặt nạ nhị phân vùng da (tùy chọn).
        color_space: 'HSV', 'YCrCb', hoặc 'RGB'.
        bins: Số khoảng chia (bins) cho histogram.
        is_bgr: True nếu ảnh đầu vào là BGR.
        save_path: Đường dẫn lưu biểu đồ (tùy chọn).

    Returns:
        plt.Figure: Đối tượng đồ họa Matplotlib.
    """
    channels_dict = split_channels(image, color_space=color_space, is_bgr=is_bgr)
    names = list(channels_dict.keys())

    # Lấy các điểm ảnh hợp lệ
    pixels_list: List[np.ndarray] = []
    for name in names:
        data = channels_dict[name]
        if mask is not None:
            pix = data[mask > 0]
        else:
            pix = data.ravel()
        pixels_list.append(pix)

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(2, 3)

    region_label = "Vùng Da Mặt (Skin Region)" if mask is not None else "Toàn Bộ Ảnh"
    fig.suptitle(f"Phân Bố Màu Sắc: {color_space.upper()} - {region_label}", fontsize=14, fontweight="bold")

    colors_palette = ["crimson", "forestgreen", "royalblue"]

    # 3 ô đầu tiên: Biểu đồ 1D cho từng kênh
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        if pixels_list[i].size > 0:
            ax.hist(pixels_list[i], bins=bins, color=colors_palette[i], alpha=0.75, edgecolor="black", linewidth=0.5)
            mu = np.mean(pixels_list[i])
            sigma = np.std(pixels_list[i])
            ax.set_title(f"{names[i]} (μ={mu:.1f}, σ={sigma:.1f})", fontsize=11, fontweight="bold")
        else:
            ax.set_title(f"{names[i]} (Không có pixel)", fontsize=11)
        ax.set_xlim([0, 256])
        ax.grid(True, linestyle="--", alpha=0.5)

    # Ô thứ 4: Phân bố 2D liên hợp giữa kênh 0 và kênh 1
    # Trong HSV là H vs S; trong YCrCb là Cr vs Cb
    ax_2d_1 = fig.add_subplot(gs[1, 0:2])
    if pixels_list[0].size > 0 and pixels_list[1].size > 0:
        h2d, xedges, yedges, im2d = ax_2d_1.hist2d(
            pixels_list[0], pixels_list[1],
            bins=bins, range=[[0, 256], [0, 256]],
            cmap="viridis"
        )
        ax_2d_1.set_xlabel(names[0], fontsize=11, fontweight="bold")
        ax_2d_1.set_ylabel(names[1], fontsize=11, fontweight="bold")
        ax_2d_1.set_title(f"Phân Bố Liên Hợp 2D ({names[0]} vs {names[1]})", fontsize=11, fontweight="bold")
        fig.colorbar(im2d, ax=ax_2d_1, fraction=0.046, pad=0.04)
    else:
        ax_2d_1.text(0.5, 0.5, "Không có dữ liệu pixel", ha="center", va="center")

    # Ô thứ 5: Phân bố 2D liên hợp giữa kênh 1 và kênh 2 (hoặc Cr vs Cb)
    ax_2d_2 = fig.add_subplot(gs[1, 2])
    if pixels_list[1].size > 0 and pixels_list[2].size > 0:
        h2d_2, _, _, im2d_2 = ax_2d_2.hist2d(
            pixels_list[1], pixels_list[2],
            bins=bins, range=[[0, 256], [0, 256]],
            cmap="plasma"
        )
        ax_2d_2.set_xlabel(names[1], fontsize=10, fontweight="bold")
        ax_2d_2.set_ylabel(names[2], fontsize=10, fontweight="bold")
        ax_2d_2.set_title(f"Phân Bố 2D ({names[1]} vs {names[2]})", fontsize=11, fontweight="bold")
        fig.colorbar(im2d_2, ax=ax_2d_2, fraction=0.046, pad=0.04)
    else:
        ax_2d_2.text(0.5, 0.5, "Không có dữ liệu pixel", ha="center", va="center")

    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200, bbox_inches="tight")

    return fig


__all__ = [
    "convert_to_hsv",
    "convert_to_ycrcb",
    "convert_to_gray",
    "convert_to_lab",
    "split_channels",
    "visualize_channels",
    "compare_color_spaces",
    "extract_color_statistics",
    "visualize_color_distribution",
]

