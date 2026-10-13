"""
Mô-đun Khám phá & Trực quan hóa Dữ liệu (Data Exploration & Visualization Module)
==============================================================================
Đề tài: Phân loại loại da mặt (Dry, Normal, Oily) sử dụng Kỹ thuật Xử lý ảnh số & Học máy.

Chức năng chính:
    1. analyze_single_image_quality: Đánh giá độ sắc nét (Laplacian variance), độ sáng, độ tương phản.
    2. analyze_dataset_quality: Phân tích chất lượng toàn bộ tập dữ liệu hoặc tập mẫu.
    3. identify_problematic_images: Phát hiện ảnh lỗi, mờ nhòe, quá tối, cháy sáng.
    4. plot_class_distribution: Vẽ biểu đồ phân bố số lượng ảnh giữa các lớp và phân tập.
    5. display_sample_images: Hiển thị lưới ảnh mẫu đại diện cho từng loại da.
    6. plot_image_size_distribution: Vẽ phân bố độ phân giải (Width x Height) và tỉ lệ khung hình.
    7. plot_quality_distributions: Trực quan hóa độ mờ, độ sáng, và tỉ lệ bóng dầu giữa các lớp.
    8. generate_exploration_report: Tự động kết xuất toàn bộ biểu đồ và báo cáo Markdown tổng kết.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# Cấu hình thiết lập giao diện đồ thị matplotlib
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

try:
    from . import config
except ImportError:
    import config


# ==============================================================================
# PHÂN TÍCH CHẤT LƯỢNG ẢNH ĐƠN LẺ (SINGLE IMAGE QUALITY ANALYSIS)
# ==============================================================================
def analyze_single_image_quality(
    image_path: Union[Path, str]
) -> Dict[str, Union[float, bool]]:
    """
    Phân tích các chỉ số chất lượng ảnh số cho một bức ảnh cụ thể:
        - Độ sắc nét / mờ: Phương sai của toán tử vi phân bậc hai Laplacian (Laplacian Variance).
          Ảnh càng sắc nét thì độ biến thiên cạnh (edge variation) càng cao.
        - Độ sáng trung bình (Mean Intensity): Tính trên kênh ảnh xám.
        - Độ tương phản (Contrast): Độ lệch chuẩn của cường độ sáng (RMS contrast).
        - Độ bão hòa màu trung bình (Mean Saturation): Trên kênh S của HSV.
        - Tỉ lệ vùng chói bóng nhờn (Specular Highlight Ratio): Pixel có V > 230 & S < 35.

    Args:
        image_path (Path | str): Đường dẫn tới tệp ảnh.

    Returns:
        Dict chứa các chỉ số định lượng và cờ cảnh báo chất lượng.
    """
    path_obj = Path(image_path)
    default_metrics = {
        "laplacian_var": 0.0,
        "brightness_mean": 0.0,
        "contrast_std": 0.0,
        "saturation_mean": 0.0,
        "highlight_ratio": 0.0,
        "is_blurry": True,
        "is_too_dark": True,
        "is_too_bright": False,
        "is_problematic": True,
    }

    if not path_obj.exists():
        return default_metrics

    try:
        # Đọc ảnh an toàn với đường dẫn tiếng Việt / khoảng trắng
        img_bgr = cv2.imdecode(
            np.fromfile(str(path_obj), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if img_bgr is None:
            return default_metrics

        # 1. Chuyển sang ảnh xám để đo độ nét và độ sáng
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Phương sai Laplacian phát hiện ảnh mờ (Blur detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())

        # Độ sáng trung bình & Độ tương phản
        brightness_mean = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        # 2. Chuyển sang HSV để đo độ bão hòa và điểm chói bóng nhờn (đặc trưng da dầu)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat_channel = hsv[:, :, 1]
        val_channel = hsv[:, :, 2]
        saturation_mean = float(np.mean(sat_channel))

        # Mặt nạ bóng sáng (Specular highlight): V cao, S thấp
        highlight_mask = (val_channel >= config.HIGHLIGHT_VALUE_THRESHOLD) & (
            sat_channel <= config.HIGHLIGHT_SAT_THRESHOLD
        )
        highlight_ratio = float(np.sum(highlight_mask) / (img_bgr.shape[0] * img_bgr.shape[1]))

        # Đánh giá theo ngưỡng
        is_blurry = laplacian_var < config.QUALITY_THRESHOLDS["blur_laplacian_var"]
        is_too_dark = brightness_mean < config.QUALITY_THRESHOLDS["dark_intensity"]
        is_too_bright = brightness_mean > config.QUALITY_THRESHOLDS["bright_intensity"]
        is_problematic = is_blurry or is_too_dark or is_too_bright

        return {
            "laplacian_var": round(laplacian_var, 2),
            "brightness_mean": round(brightness_mean, 2),
            "contrast_std": round(contrast_std, 2),
            "saturation_mean": round(saturation_mean, 2),
            "highlight_ratio": round(highlight_ratio, 4),
            "is_blurry": is_blurry,
            "is_too_dark": is_too_dark,
            "is_too_bright": is_too_bright,
            "is_problematic": is_problematic,
        }

    except Exception:
        return default_metrics


# ==============================================================================
# PHÂN TÍCH CHẤT LƯỢNG TOÀN BỘ DATASET (DATASET QUALITY ANALYSIS)
# ==============================================================================
def analyze_dataset_quality(
    df: pd.DataFrame, max_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Phân tích chất lượng ảnh cho toàn bộ bảng dữ liệu hoặc một tập con.
    Enriches dataset DataFrame with image quality metrics.

    Args:
        df (pd.DataFrame): DataFrame chứa cột 'image_path'.
        max_samples (Optional[int]): Giới hạn số lượng ảnh nếu tập dữ liệu quá lớn.

    Returns:
        pd.DataFrame: DataFrame mới đã bổ sung các cột chỉ số chất lượng ảnh.
    """
    df_copy = df.copy()
    if len(df_copy) == 0:
        return df_copy

    if max_samples and max_samples < len(df_copy):
        target_df = df_copy.sample(n=max_samples, random_state=config.RANDOM_STATE)
    else:
        target_df = df_copy

    quality_results = []
    for _, row in target_df.iterrows():
        # Chỉ phân tích nếu ảnh hợp lệ
        if row.get("is_valid", True):
            q_res = analyze_single_image_quality(row["image_path"])
        else:
            q_res = {
                "laplacian_var": 0.0,
                "brightness_mean": 0.0,
                "contrast_std": 0.0,
                "saturation_mean": 0.0,
                "highlight_ratio": 0.0,
                "is_blurry": True,
                "is_too_dark": False,
                "is_too_bright": False,
                "is_problematic": True,
            }
        quality_results.append(q_res)

    quality_df = pd.DataFrame(quality_results, index=target_df.index)

    for col in quality_df.columns:
        df_copy[col] = quality_df[col]

    return df_copy


# ==============================================================================
# NHẬN DIỆN ẢNH CÓ VẤN ĐỀ (IDENTIFY PROBLEMATIC IMAGES)
# ==============================================================================
def identify_problematic_images(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Phát hiện và thống kê các ảnh tiềm ẩn rủi ro về chất lượng (quá mờ, quá tối, cháy sáng).
    Identifies low-quality or corrupted images.

    Args:
        df (pd.DataFrame): DataFrame đã được tính toán chỉ số chất lượng.

    Returns:
        Tuple:
            - problematic_df (pd.DataFrame): Danh sách các hàng ảnh cần lưu ý.
            - summary (Dict[str, Any]): Báo cáo đếm số lượng lỗi theo từng tiêu chí.
    """
    if len(df) == 0:
        return pd.DataFrame(), {}

    cond_invalid = ~df.get("is_valid", pd.Series([True] * len(df), index=df.index))
    cond_blurry = df.get("is_blurry", pd.Series([False] * len(df), index=df.index))
    cond_dark = df.get("is_too_dark", pd.Series([False] * len(df), index=df.index))
    cond_bright = df.get("is_too_bright", pd.Series([False] * len(df), index=df.index))

    problem_mask = cond_invalid | cond_blurry | cond_dark | cond_bright
    problematic_df = df[problem_mask].copy()

    # Thêm lý do tổng hợp
    def get_reason(r):
        reasons = []
        if not r.get("is_valid", True):
            reasons.append(f"Invalid: {r.get('error_reason', 'Corrupted')}")
        if r.get("is_blurry", False):
            reasons.append(f"Blurry (Laplacian={r.get('laplacian_var', 0)})")
        if r.get("is_too_dark", False):
            reasons.append(f"Too Dark (Mean={r.get('brightness_mean', 0)})")
        if r.get("is_too_bright", False):
            reasons.append(f"Too Bright (Mean={r.get('brightness_mean', 0)})")
        return "; ".join(reasons)

    problematic_df["problem_reasons"] = problematic_df.apply(get_reason, axis=1)

    summary = {
        "total_problematic": int(len(problematic_df)),
        "corrupted_or_invalid": int(cond_invalid.sum()),
        "blurry_count": int(cond_blurry.sum()),
        "too_dark_count": int(cond_dark.sum()),
        "too_bright_count": int(cond_bright.sum()),
    }

    return problematic_df, summary


# ==============================================================================
# VẼ BIỂU ĐỒ PHÂN BỐ LỚP & TẬP DỮ LIỆU (CLASS DISTRIBUTION PLOT)
# ==============================================================================
def plot_class_distribution(
    df: pd.DataFrame,
    save_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """
    Vẽ biểu đồ phân bố số lượng ảnh giữa các loại da (Dry, Normal, Oily)
    và phân bố chi tiết trên các tập Train / Valid / Test.
    Plots class balance and split proportions.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        save_path (Optional[Path | str]): Đường dẫn lưu ảnh PNG.

    Returns:
        plt.Figure: Đối tượng đồ thị matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    palette = [config.CLASS_COLORS.get(c, "#333333") for c in config.CLASS_NAMES]

    # Subplot 1: Tổng thể các lớp (Overall Class Distribution)
    ax1 = axes[0]
    class_counts = df["class_name"].value_counts().reindex(config.CLASS_NAMES, fill_value=0)
    total_imgs = len(df)

    bars = ax1.bar(
        [config.CLASS_LABELS_VI.get(c, c) for c in class_counts.index],
        class_counts.values,
        color=palette,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.85,
    )
    ax1.set_title("Phân bố Tổng thể các Loại Da (Class Distribution)", fontsize=13, weight="bold")
    ax1.set_ylabel("Số lượng ảnh (Image Count)", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Ghi nhãn số lượng và phần trăm trên từng cột
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_imgs * 100) if total_imgs > 0 else 0
        ax1.annotate(
            f"{int(h)}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    # Subplot 2: Phân bố theo Split (Train, Valid, Test)
    ax2 = axes[1]
    if "split" in df.columns and len(df["split"].unique()) > 1:
        split_order = [s for s in ["train", "valid", "test"] if s in df["split"].unique()]
        ctab = pd.crosstab(df["split"], df["class_name"]).reindex(split_order)
        ctab = ctab.reindex(columns=config.CLASS_NAMES, fill_value=0)

        ctab.plot(
            kind="bar",
            ax=ax2,
            color=palette,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
            width=0.7,
        )
        ax2.set_title("Phân bố Lớp theo từng Tập Dữ liệu (Train/Val/Test)", fontsize=13, weight="bold")
        ax2.set_xlabel("Phân tập dữ liệu (Dataset Split)", fontsize=11)
        ax2.set_ylabel("Số lượng ảnh", fontsize=11)
        ax2.legend(
            [config.CLASS_LABELS_VI.get(c, c) for c in config.CLASS_NAMES],
            title="Loại da",
            frameon=True,
        )
        ax2.set_xticklabels([s.capitalize() for s in split_order], rotation=0)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)
    else:
        ax2.text(
            0.5,
            0.5,
            "Chưa phân chia Train/Valid/Test\n(Dataset is not partitioned yet)",
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
        )
        ax2.set_axis_off()

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


# ==============================================================================
# HIỂN THỊ LƯỚI ẢNH MẪU ĐẠI DIỆN (SAMPLE IMAGES GRID)
# ==============================================================================
def display_sample_images(
    df: pd.DataFrame,
    n_samples_per_class: int = 4,
    save_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """
    Trực quan hóa lưới ảnh mẫu đại diện cho 3 lớp da (Dry, Normal, Oily).
    Displays a visual grid of sample facial images per class.

    Args:
        df (pd.DataFrame): DataFrame dữ liệu.
        n_samples_per_class (int): Số ảnh mẫu trên mỗi lớp (mặc định 4 ảnh/hàng).
        save_path (Optional[Path | str]): Đường dẫn lưu tệp hình ảnh.

    Returns:
        plt.Figure: Đối tượng đồ thị matplotlib.
    """
    classes = [c for c in config.CLASS_NAMES if c in df["class_name"].unique()]
    n_classes = len(classes)

    if n_classes == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Không có dữ liệu ảnh để hiển thị", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(
        n_classes,
        n_samples_per_class,
        figsize=(n_samples_per_class * 3.2, n_classes * 3.4),
    )

    # Đảm bảo axes luôn ở dạng mảng 2 chiều
    if n_classes == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_samples_per_class == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, c_name in enumerate(classes):
        class_sub = df[df["class_name"] == c_name]
        valid_sub = class_sub[class_sub.get("is_valid", True)]
        if len(valid_sub) == 0:
            valid_sub = class_sub

        sample_rows = valid_sub.sample(
            n=min(n_samples_per_class, len(valid_sub)),
            random_state=config.RANDOM_STATE,
        )

        for col_idx in range(n_samples_per_class):
            ax = axes[row_idx, col_idx]

            if col_idx < len(sample_rows):
                row_data = sample_rows.iloc[col_idx]
                img_p = Path(row_data["image_path"])

                try:
                    # Đọc ảnh chuyển hệ BGR -> RGB để Matplotlib vẽ đúng màu
                    img_bgr = cv2.imdecode(
                        np.fromfile(str(img_p), dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        ax.imshow(img_rgb)
                        w = row_data.get("width", img_rgb.shape[1])
                        h = row_data.get("height", img_rgb.shape[0])
                        ax.set_title(
                            f"{row_data['filename'][:15]}\n({w}x{h})",
                            fontsize=9,
                        )
                    else:
                        ax.text(0.5, 0.5, "Lỗi đọc ảnh", ha="center", va="center")
                except Exception:
                    ax.text(0.5, 0.5, "Lỗi hiển thị", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")

            ax.axis("off")

        # Gắn nhãn phân loại ở cột đầu tiên
        axes[row_idx, 0].text(
            -0.15,
            0.5,
            config.CLASS_LABELS_VI.get(c_name, c_name.upper()),
            transform=axes[row_idx, 0].transAxes,
            va="center",
            ha="right",
            fontsize=12,
            weight="bold",
            color=config.CLASS_COLORS.get(c_name, "black"),
            rotation=90,
        )

    plt.suptitle("Mẫu Ảnh Đại Diện Các Loại Da Mặt (Sample Face Images by Class)", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


# ==============================================================================
# VẼ BIỂU ĐỒ PHÂN BỐ ĐỘ PHÂN GIẢI VÀ TỈ LỆ KHUNG HÌNH (IMAGE SIZE DISTRIBUTION)
# ==============================================================================
def plot_image_size_distribution(
    df: pd.DataFrame,
    save_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """
    Vẽ phân bố độ phân giải (Width vs Height), tỉ lệ khung hình (Aspect Ratio),
    và dung lượng tệp ảnh.
    Plots image dimensions, aspect ratios, and file sizes.

    Args:
        df (pd.DataFrame): DataFrame chứa thông tin width, height, aspect_ratio, file_size_kb.
        save_path (Optional[Path | str]): Đường dẫn lưu hình.

    Returns:
        plt.Figure: Đối tượng đồ thị matplotlib.
    """
    valid_df = df[df.get("is_valid", True)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Biểu đồ tán xạ (Scatter Plot): Chiều rộng vs Chiều cao
    ax1 = axes[0]
    for c_name in config.CLASS_NAMES:
        sub = valid_df[valid_df["class_name"] == c_name]
        if len(sub) > 0:
            ax1.scatter(
                sub["width"],
                sub["height"],
                label=config.CLASS_LABELS_VI.get(c_name, c_name),
                color=config.CLASS_COLORS.get(c_name, "#666666"),
                alpha=0.6,
                s=28,
                edgecolors="none",
            )
    # Vẽ điểm mục tiêu chuẩn hóa Target Size (224x224)
    ax1.axvline(config.TARGET_SIZE[0], color="red", linestyle="--", linewidth=1.5, label=f"Target Width ({config.TARGET_SIZE[0]})")
    ax1.axhline(config.TARGET_SIZE[1], color="red", linestyle=":", linewidth=1.5, label=f"Target Height ({config.TARGET_SIZE[1]})")

    ax1.set_title("Độ phân giải Gốc (Width vs Height)", fontsize=12, weight="bold")
    ax1.set_xlabel("Chiều rộng (Pixels)", fontsize=10)
    ax1.set_ylabel("Chiều cao (Pixels)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Histogram tỉ lệ khung hình (Aspect Ratio = Width / Height)
    ax2 = axes[1]
    if "aspect_ratio" in valid_df.columns:
        sns.histplot(
            valid_df["aspect_ratio"],
            kde=True,
            ax=ax2,
            color="#2B5B84",
            bins=25,
            edgecolor="black",
        )
        ax2.axvline(1.0, color="orange", linestyle="--", linewidth=1.5, label="Vuông (1:1)")
        ax2.set_title("Phân bố Tỉ lệ Khung hình (Aspect Ratio = W/H)", fontsize=12, weight="bold")
        ax2.set_xlabel("Tỉ lệ W/H", fontsize=10)
        ax2.set_ylabel("Tần suất (Count)", fontsize=10)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Boxplot dung lượng tệp (File Size in KB) theo lớp
    ax3 = axes[2]
    if "file_size_kb" in valid_df.columns:
        box_data = [
            valid_df[valid_df["class_name"] == c]["file_size_kb"].dropna()
            for c in config.CLASS_NAMES
            if len(valid_df[valid_df["class_name"] == c]) > 0
        ]
        labels = [config.CLASS_LABELS_VI.get(c, c) for c in config.CLASS_NAMES if len(valid_df[valid_df["class_name"] == c]) > 0]
        ax3.boxplot(box_data, patch_artist=True, labels=labels)
        ax3.set_title("Phân bố Dung lượng Tệp (File Size KB)", fontsize=12, weight="bold")
        ax3.set_ylabel("Dung lượng (KB)", fontsize=10)
        ax3.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


# ==============================================================================
# VẼ BIỂU ĐỒ CHỈ SỐ CHẤT LƯỢNG ẢNH (IMAGE QUALITY DISTRIBUTIONS)
# ==============================================================================
def plot_quality_distributions(
    df: pd.DataFrame,
    save_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """
    Vẽ phân bố các chỉ số chất lượng ảnh quan trọng:
        1. Độ sắc nét (Laplacian Variance - Blur Detection).
        2. Độ sáng trung bình (Grayscale Intensity - Brightness).
        3. Tỉ lệ vùng chói bóng nhờn (Specular Highlight Ratio - Oiliness Indicator).
        4. Độ tương phản ảnh (Grayscale Standard Deviation).
    Plots quality and artifact distributions across skin classes.

    Args:
        df (pd.DataFrame): DataFrame đã có các cột chất lượng ảnh.
        save_path (Optional[Path | str]): Đường dẫn lưu đồ thị.

    Returns:
        plt.Figure: Đối tượng đồ thị matplotlib.
    """
    valid_df = df[df.get("is_valid", True)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: Độ sắc nét (Laplacian Variance)
    ax1 = axes[0, 0]
    if "laplacian_var" in valid_df.columns:
        for c in config.CLASS_NAMES:
            sub = valid_df[valid_df["class_name"] == c]["laplacian_var"].dropna()
            if len(sub) > 0:
                sns.kdeplot(
                    sub,
                    ax=ax1,
                    label=config.CLASS_LABELS_VI.get(c, c),
                    color=config.CLASS_COLORS.get(c, "blue"),
                    fill=True,
                    alpha=0.25,
                )
        blur_thresh = config.QUALITY_THRESHOLDS["blur_laplacian_var"]
        ax1.axvline(blur_thresh, color="red", linestyle="--", linewidth=1.5, label=f"Ngưỡng mờ ({blur_thresh})")
        ax1.set_title("Phân bố Độ sắc nét (Laplacian Variance)", fontsize=12, weight="bold")
        ax1.set_xlabel("Toán tử Laplacian Variance (Càng cao càng nét)", fontsize=10)
        ax1.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Độ sáng trung bình (Brightness Intensity)
    ax2 = axes[0, 1]
    if "brightness_mean" in valid_df.columns:
        for c in config.CLASS_NAMES:
            sub = valid_df[valid_df["class_name"] == c]["brightness_mean"].dropna()
            if len(sub) > 0:
                sns.kdeplot(
                    sub,
                    ax=ax2,
                    label=config.CLASS_LABELS_VI.get(c, c),
                    color=config.CLASS_COLORS.get(c, "blue"),
                    fill=True,
                    alpha=0.25,
                )
        dark_thresh = config.QUALITY_THRESHOLDS["dark_intensity"]
        bright_thresh = config.QUALITY_THRESHOLDS["bright_intensity"]
        ax2.axvline(dark_thresh, color="black", linestyle="--", label=f"Quá tối (<{dark_thresh})")
        ax2.axvline(bright_thresh, color="orange", linestyle="--", label=f"Quá sáng (>{bright_thresh})")
        ax2.set_title("Phân bố Độ sáng Trung bình (Grayscale Brightness)", fontsize=12, weight="bold")
        ax2.set_xlabel("Độ sáng [0 - 255]", fontsize=10)
        ax2.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Tỉ lệ vùng chói bóng nhờn (Highlight Ratio)
    ax3 = axes[1, 0]
    if "highlight_ratio" in valid_df.columns:
        for c in config.CLASS_NAMES:
            sub = valid_df[valid_df["class_name"] == c]["highlight_ratio"].dropna()
            if len(sub) > 0:
                sns.kdeplot(
                    sub,
                    ax=ax3,
                    label=config.CLASS_LABELS_VI.get(c, c),
                    color=config.CLASS_COLORS.get(c, "blue"),
                    fill=True,
                    alpha=0.25,
                )
        ax3.set_title("Tỉ lệ Điểm chói bóng nhờn (Specular Highlight Ratio)", fontsize=12, weight="bold")
        ax3.set_xlabel("Tỉ lệ pixel chói (V>230, S<35)", fontsize=10)
        ax3.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Độ tương phản (Contrast - Standard Deviation)
    ax4 = axes[1, 1]
    if "contrast_std" in valid_df.columns:
        for c in config.CLASS_NAMES:
            sub = valid_df[valid_df["class_name"] == c]["contrast_std"].dropna()
            if len(sub) > 0:
                sns.kdeplot(
                    sub,
                    ax=ax4,
                    label=config.CLASS_LABELS_VI.get(c, c),
                    color=config.CLASS_COLORS.get(c, "blue"),
                    fill=True,
                    alpha=0.25,
                )
        ax4.set_title("Phân bố Độ tương phản (Grayscale Std Contrast)", fontsize=12, weight="bold")
        ax4.set_xlabel("Độ lệch chuẩn mức xám", fontsize=10)
        ax4.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


# ==============================================================================
# TẠO BÁO CÁO THÁM HIỂM TOÀN DIỆN (FULL EXPLORATION REPORT)
# ==============================================================================
def generate_exploration_report(
    dataset_info: Any,
    output_dir: Optional[Union[Path, str]] = None,
    sample_quality_size: Optional[int] = 500,
) -> Dict[str, Any]:
    """
    Tự động chạy toàn bộ chu trình thám hiểm dữ liệu:
        1. Phân tích chất lượng các ảnh trong dataset.
        2. Nhận diện các ảnh có vấn đề.
        3. Kết xuất và lưu trữ tất cả các biểu đồ vào thư mục outputs/figures/.
        4. Xuất danh sách ảnh lỗi ra outputs/reports/problematic_images.csv.
        5. Xuất báo cáo tổng hợp Markdown ra outputs/reports/data_exploration_report.md.

    Args:
        dataset_info (Any): Đối tượng DatasetInfo từ data_loader.
        output_dir (Optional[Path | str]): Thư mục lưu báo cáo. Mặc định là config.OUTPUTS_DIR.
        sample_quality_size (Optional[int]): Kích thước mẫu tính toán chất lượng nếu tập dữ liệu lớn.

    Returns:
        Dict[str, Any]: Báo cáo tóm tắt các đường dẫn tệp và chỉ số chất lượng.
    """
    out_dir = Path(output_dir) if output_dir else config.OUTPUTS_DIR
    fig_dir = out_dir / "figures"
    rep_dir = out_dir / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    df = dataset_info.df

    # 1. Phân tích chất lượng ảnh
    df_quality = analyze_dataset_quality(df, max_samples=sample_quality_size)
    dataset_info.df = df_quality

    # 2. Nhận diện các ảnh có vấn đề
    prob_df, prob_summary = identify_problematic_images(df_quality)

    # 3. Xuất file problematic_images.csv
    prob_csv_path = rep_dir / "problematic_images.csv"
    if len(prob_df) > 0:
        prob_df.to_csv(prob_csv_path, index=False, encoding="utf-8")

    # 4. Vẽ các biểu đồ
    plot1_path = fig_dir / "class_distribution.png"
    plot_class_distribution(df_quality, save_path=plot1_path)
    plt.close()

    plot2_path = fig_dir / "sample_images_grid.png"
    display_sample_images(df_quality, n_samples_per_class=4, save_path=plot2_path)
    plt.close()

    plot3_path = fig_dir / "image_size_distribution.png"
    plot_image_size_distribution(df_quality, save_path=plot3_path)
    plt.close()

    plot4_path = fig_dir / "image_quality_distribution.png"
    plot_quality_distributions(df_quality, save_path=plot4_path)
    plt.close()

    # 5. Soạn thảo báo cáo Markdown tổng hợp
    stats = dataset_info.statistics
    report_md_path = rep_dir / "data_exploration_report.md"

    classes_table_rows = []
    for c in config.CLASS_NAMES:
        c_cnt = stats.get("classes_counts", {}).get(c, 0)
        c_pct = stats.get("classes_percentages", {}).get(c, 0.0)
        classes_table_rows.append(f"| {config.CLASS_LABELS_VI.get(c, c)} | {c_cnt} | {c_pct}% |")
    classes_table = "\n".join(classes_table_rows)

    splits_table_rows = []
    for s, cnt in stats.get("splits_counts", {}).items():
        splits_table_rows.append(f"| {s.capitalize()} | {cnt} |")
    splits_table = "\n".join(splits_table_rows) if splits_table_rows else "| Không xác định | 0 |"

    md_content = f"""# BÁO CÁO THÁM HIỂM VÀ KIỂM ĐỊNH CHẤT LƯỢNG BỘ DỮ LIỆU
## Đồ án Xử lý Ảnh số: Phân loại loại da mặt (Dry, Normal, Oily)

---

### 1. Tổng quan Bộ dữ liệu
- **Thư mục nguồn:** `{dataset_info.root_path}`
- **Loại cấu trúc phát hiện:** `{dataset_info.structure_type}`
- **Tổng số ảnh thu thập:** `{stats.get('total_images', 0)}`
- **Số ảnh hợp lệ (Valid):** `{stats.get('valid_images', 0)}`
- **Số ảnh lỗi không đọc được (Corrupted):** `{stats.get('invalid_images', 0)}`

### 2. Phân bố Lớp (Class Distribution)
| Loại da | Số lượng ảnh | Tỉ lệ (%) |
| :--- | :---: | :---: |
{classes_table}

### 3. Phân bố theo Phân tập (Splits)
| Phân tập | Số lượng ảnh |
| :--- | :---: |
{splits_table}

### 4. Thống kê Kích thước & Độ phân giải (Dimensions)
- **Chiều rộng (Width):** Min = {stats.get('dimensions', {}).get('width_min', 'N/A')}, Max = {stats.get('dimensions', {}).get('width_max', 'N/A')}, Trung bình = {stats.get('dimensions', {}).get('width_mean', 'N/A')} px
- **Chiều cao (Height):** Min = {stats.get('dimensions', {}).get('height_min', 'N/A')}, Max = {stats.get('dimensions', {}).get('height_max', 'N/A')}, Trung bình = {stats.get('dimensions', {}).get('height_mean', 'N/A')} px
- **Tỉ lệ khung hình trung bình (Aspect Ratio W/H):** {stats.get('dimensions', {}).get('aspect_ratio_mean', 'N/A')}
- **Tổng dung lượng bộ dữ liệu:** {stats.get('file_size_kb', {}).get('total_mb', 'N/A')} MB

### 5. Kiểm tra Trùng lặp & Chất lượng Ảnh
- **Ảnh trùng lặp tuyệt đối (MD5 duplicate):** {dataset_info.duplicates_summary.get('exact_duplicates_count', 0)}
- **Ảnh tương đồng cảm nhận (Near dHash duplicate):** {dataset_info.duplicates_summary.get('near_duplicates_count', 0)}
- **Số ảnh bị mờ nét (Laplacian Var < {config.QUALITY_THRESHOLDS['blur_laplacian_var']}):** {prob_summary.get('blurry_count', 0)}
- **Số ảnh quá tối (Brightness < {config.QUALITY_THRESHOLDS['dark_intensity']}):** {prob_summary.get('too_dark_count', 0)}
- **Số ảnh quá sáng (Brightness > {config.QUALITY_THRESHOLDS['bright_intensity']}):** {prob_summary.get('too_bright_count', 0)}
- **Tổng số ảnh cảnh báo chất lượng:** {prob_summary.get('total_problematic', 0)}

---

### 6. Danh mục Biểu đồ Đã Kết xuất
- `figures/class_distribution.png`: Biểu đồ cân bằng lớp và phân tập dữ liệu.
- `figures/sample_images_grid.png`: Lưới mẫu khuôn mặt đại diện theo từng loại da.
- `figures/image_size_distribution.png`: Tán xạ độ phân giải và phân bố tỉ lệ khung hình.
- `figures/image_quality_distribution.png`: Phân bố độ sắc nét, độ sáng và vùng chói bóng nhờn.
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return {
        "report_md_path": str(report_md_path),
        "problematic_csv_path": str(prob_csv_path) if len(prob_df) > 0 else None,
        "figures": [
            str(plot1_path),
            str(plot2_path),
            str(plot3_path),
            str(plot4_path),
        ],
        "problematic_summary": prob_summary,
    }
