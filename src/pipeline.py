"""
Module Đường Ống Xử Lý Toàn Diện (Complete Processing Pipeline Module)
Dự án: Phân loại loại da mặt (Xử lý ảnh số - Digital Image Processing)

File: src/pipeline.py
Mục đích:
- Xây dựng 2 đường ống xử lý hoàn chỉnh (End-to-End Pipelines):
    1. BaselinePipeline:
       Đọc ảnh -> Resize trực tiếp -> Chuyển không gian màu -> Phân vùng da cơ bản (ngưỡng HSV/YCrCb) -> Trích xuất đặc trưng.
    2. EnhancedPipeline:
       Đọc ảnh -> Resize -> Khử nhiễu (Gaussian/Median) -> Cân bằng tương phản thích ứng (CLAHE) ->
       Phân vùng da nâng cao (kết hợp đa không gian màu + hình thái học đóng/mở + lấp lỗ hổng) -> Trích xuất đặc trưng.
- Lưu trữ toàn bộ kết quả trung gian để minh họa bài giảng và báo cáo đồ án.
- Đo đạc chính xác thời gian thực thi (latency ms) từng công đoạn trong chuỗi xử lý.
- Cung cấp hàm xử lý ảnh đơn lẻ (phục vụ demo / web app) và hàm xử lý hàng loạt theo lô (batch dataset processing).
- Cung cấp hàm so sánh định lượng và trực quan giữa Baseline vs Enhanced Pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Nhập các module thành phần trong kiến trúc src
try:
    from src import config
    from src.color_space import convert_to_hsv, convert_to_ycrcb
    from src.feature_extraction import extract_features, get_feature_names
    from src.preprocessing import (
        apply_clahe,
        apply_gaussian_blur,
        apply_median_blur,
        bgr_to_rgb,
        read_image,
        resize_image,
    )
    from src.skin_detection import (
        baseline_skin_detection,
        enhanced_skin_detection,
        extract_roi,
    )
    from src.utils import Timer, format_duration
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.color_space import convert_to_hsv, convert_to_ycrcb
    from src.feature_extraction import extract_features, get_feature_names
    from src.preprocessing import (
        apply_clahe,
        apply_gaussian_blur,
        apply_median_blur,
        bgr_to_rgb,
        read_image,
        resize_image,
    )
    from src.skin_detection import (
        baseline_skin_detection,
        enhanced_skin_detection,
        extract_roi,
    )
    from src.utils import Timer, format_duration

logger = logging.getLogger("xla_pipeline.pipeline")


# ==============================================================================
# 1. BASELINE PIPELINE (ĐƯỜNG ỐNG XỬ LÝ ẢNH CƠ BẢN)
# ==============================================================================

class BaselinePipeline:
    """
    Đường ống xử lý cơ bản (Baseline Pipeline):
    Quy trình xử lý tuần tự truyền thống:
        1. Đọc ảnh (Read image).
        2. Thay đổi kích thước trực tiếp về TARGET_SIZE (224x224).
        3. Chuyển đổi không gian màu (BGR -> RGB/HSV/YCbCr).
        4. Phân đoạn sắc tố da bằng ngưỡng màu đơn giản (Simple HSV thresholding).
        5. Trích xuất vector đặc trưng trên vùng da.

    Lưu trữ toàn bộ ảnh trung gian và thống kê thời gian từng bước để so sánh.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        color_space: str = "RGB",
        feature_group: str = "all",
        hist_bins: int = 32,
        k_dominant: int = 1,
        lbp_radius: int = 1,
        lbp_n_points: int = 8,
    ) -> None:
        """
        Khởi tạo BaselinePipeline với các tham số cấu hình.
        """
        self.target_size = target_size
        self.color_space = color_space.upper()
        self.feature_group = feature_group
        self.hist_bins = hist_bins
        self.k_dominant = k_dominant
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points

    def process(
        self,
        image_input: Union[str, Path, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Thực thi toàn bộ chuỗi xử lý Baseline trên một ảnh.
        Execute complete baseline pipeline on a single image.

        Args:
            image_input: Đường dẫn tệp ảnh hoặc mảng numpy ảnh BGR.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - features: Vector đặc trưng 1D (np.float32).
                - metadata: Từ điển chứa ảnh trung gian, thông số chất lượng và thời gian thực thi.
        """
        timings: Dict[str, float] = {}
        intermediates: Dict[str, Any] = {}
        t_total_start = time.perf_counter()

        # 1. Đọc và hợp thức hóa ảnh
        t0 = time.perf_counter()
        raw_bgr = read_image(image_input, as_rgb=False)
        timings["read_ms"] = (time.perf_counter() - t0) * 1000.0
        intermediates["raw_bgr"] = raw_bgr
        intermediates["raw_rgb"] = bgr_to_rgb(raw_bgr)

        # 2. Resize kích thước trực tiếp (không letterbox)
        t0 = time.perf_counter()
        resized_bgr, _ = resize_image(raw_bgr, target_size=self.target_size, keep_aspect_ratio=False)
        timings["resize_ms"] = (time.perf_counter() - t0) * 1000.0
        intermediates["resized_bgr"] = resized_bgr
        intermediates["resized_rgb"] = bgr_to_rgb(resized_bgr)

        # 3. Phân vùng da theo phương pháp cơ bản (Ngưỡng HSV thuần túy)
        t0 = time.perf_counter()
        skin_mask, extracted_skin, skin_meta = baseline_skin_detection(
            resized_bgr, color_space="hsv", is_bgr=True
        )
        timings["skin_detect_ms"] = (time.perf_counter() - t0) * 1000.0
        intermediates["skin_mask"] = skin_mask
        intermediates["extracted_skin_bgr"] = extracted_skin
        intermediates["extracted_skin_rgb"] = bgr_to_rgb(extracted_skin)

        # 4. Trích xuất đặc trưng trên vùng da
        t0 = time.perf_counter()
        features, feature_names, feat_meta = extract_features(
            image=resized_bgr,
            mask=skin_mask,
            feature_group=self.feature_group,
            color_space=self.color_space,
            hist_bins=self.hist_bins,
            k_dominant=self.k_dominant,
            lbp_radius=self.lbp_radius,
            lbp_n_points=self.lbp_n_points,
            is_bgr=True
        )
        timings["feature_extract_ms"] = (time.perf_counter() - t0) * 1000.0

        total_time_ms = (time.perf_counter() - t_total_start) * 1000.0
        timings["total_ms"] = total_time_ms

        metadata: Dict[str, Any] = {
            "pipeline_type": "baseline",
            "timings_ms": timings,
            "intermediates": intermediates,
            "skin_quality": skin_meta.get("quality", {}),
            "feature_info": feat_meta,
            "feature_names": feature_names,
            "feature_dim": len(features),
        }
        return features, metadata


# ==============================================================================
# 2. ENHANCED PIPELINE (ĐƯỜNG ỐNG XỬ LÝ ẢNH NÂNG CAO)
# ==============================================================================

class EnhancedPipeline:
    """
    Đường ống xử lý nâng cao (Enhanced Pipeline):
    Tích hợp các thuật toán xử lý ảnh số hiện đại nhằm tối ưu chất lượng phân loại:
        1. Đọc và xác thực ảnh (Input validation).
        2. Thay đổi kích thước ảnh (Preserve aspect ratio / Direct resize).
        3. Khử nhiễu làm mịn thích ứng (Gaussian hoặc Median filter).
        4. Tăng cường độ tương phản cục bộ CLAHE trên kênh độ sáng (Luminance).
        5. Phân đoạn da nâng cao: Ngưỡng phối hợp đa không gian màu (HSV + YCrCb)
           kết hợp biến đổi hình thái học (Opening, Closing, FloodFill hole filling,
           Connected Component Filtering).
        6. Trích xuất toàn diện các nhóm đặc trưng A (Màu), B (Độ sáng), C (LBP Kết cấu).

    Lưu trữ đầy đủ các ma trận trung gian phục vụ phân tích trực quan sâu.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        keep_aspect_ratio: bool = False,
        denoise_method: str = "gaussian",
        gaussian_kernel: Tuple[int, int] = (5, 5),
        gaussian_sigma: float = 1.0,
        contrast_method: str = "clahe",
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        skin_detect_method: str = "combined",
        color_space: str = "RGB",
        feature_group: str = "all",
        hist_bins: int = 32,
        k_dominant: int = 1,
        lbp_radius: int = 1,
        lbp_n_points: int = 8,
    ) -> None:
        """
        Khởi tạo EnhancedPipeline với đầy đủ tham số tinh chỉnh.
        """
        self.target_size = target_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.denoise_method = denoise_method.lower()
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma
        self.contrast_method = contrast_method.lower()
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.skin_detect_method = skin_detect_method.lower()
        self.color_space = color_space.upper()
        self.feature_group = feature_group
        self.hist_bins = hist_bins
        self.k_dominant = k_dominant
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points

    def process(
        self,
        image_input: Union[str, Path, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Thực thi toàn bộ chuỗi xử lý Enhanced trên một ảnh.
        Execute complete enhanced pipeline on a single image.

        Args:
            image_input: Đường dẫn tệp ảnh hoặc mảng numpy ảnh BGR.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - features: Vector đặc trưng 1D (np.float32).
                - metadata: Từ điển chứa kết quả trung gian và thời gian từng bước.
        """
        timings: Dict[str, float] = {}
        intermediates: Dict[str, Any] = {}
        t_total_start = time.perf_counter()

        # 1. Đọc và kiểm định tính toàn vẹn của ảnh
        t0 = time.perf_counter()
        raw_bgr = read_image(image_input, as_rgb=False)
        timings["read_ms"] = (time.perf_counter() - t0) * 1000.0
        intermediates["raw_bgr"] = raw_bgr
        intermediates["raw_rgb"] = bgr_to_rgb(raw_bgr)

        # 2. Thay đổi kích thước chuẩn hóa
        t0 = time.perf_counter()
        resized_bgr, resize_meta = resize_image(
            raw_bgr,
            target_size=self.target_size,
            keep_aspect_ratio=self.keep_aspect_ratio
        )
        timings["resize_ms"] = (time.perf_counter() - t0) * 1000.0
        intermediates["resized_bgr"] = resized_bgr
        intermediates["resized_rgb"] = bgr_to_rgb(resized_bgr)
        intermediates["resize_meta"] = resize_meta

        # 3. Khử nhiễu làm mịn bề mặt
        t0 = time.perf_counter()
        if self.denoise_method == "gaussian":
            denoised_bgr, _ = apply_gaussian_blur(
                resized_bgr, kernel_size=self.gaussian_kernel, sigma=self.gaussian_sigma
            )
        elif self.denoise_method == "median":
            denoised_bgr, _ = apply_median_blur(resized_bgr, kernel_size=5)
        else:
            denoised_bgr = resized_bgr.copy()
        timings["denoise_ms"] = (time.perf_counter() - t0) * 1000.0
        intermediates["denoised_bgr"] = denoised_bgr
        intermediates["denoised_rgb"] = bgr_to_rgb(denoised_bgr)

        # 4. Tăng cường độ tương phản thích ứng (CLAHE trên kênh độ sáng L)
        t0 = time.perf_counter()
        if self.contrast_method == "clahe":
            enhanced_bgr, _ = apply_clahe(
                denoised_bgr,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_grid,
                color_space="LAB",
                is_bgr=True
            )
        else:
            enhanced_bgr = denoised_bgr.copy()
        timings["clahe_ms"] = (time.perf_counter() - t0) * 1000.0
        intermediates["enhanced_bgr"] = enhanced_bgr
        intermediates["enhanced_rgb"] = bgr_to_rgb(enhanced_bgr)

        # 5. Phân vùng da nâng cao với xử lý hình thái học & lấp lỗ hổng
        t0 = time.perf_counter()
        cleaned_mask, extracted_skin, skin_meta = enhanced_skin_detection(
            enhanced_bgr,
            method=self.skin_detect_method,
            min_region_area=150,
            fill_holes=True,
            crop_to_bbox=False,
            is_bgr=True
        )
        timings["skin_detect_ms"] = (time.perf_counter() - t0) * 1000.0
        intermediates["raw_mask"] = skin_meta.get("raw_mask", cleaned_mask)
        intermediates["cleaned_mask"] = cleaned_mask
        intermediates["extracted_skin_bgr"] = extracted_skin
        intermediates["extracted_skin_rgb"] = bgr_to_rgb(extracted_skin)

        # 6. Trích xuất đặc trưng (Feature Extraction)
        t0 = time.perf_counter()
        features, feature_names, feat_meta = extract_features(
            image=enhanced_bgr,
            mask=cleaned_mask,
            feature_group=self.feature_group,
            color_space=self.color_space,
            hist_bins=self.hist_bins,
            k_dominant=self.k_dominant,
            lbp_radius=self.lbp_radius,
            lbp_n_points=self.lbp_n_points,
            is_bgr=True
        )
        timings["feature_extract_ms"] = (time.perf_counter() - t0) * 1000.0

        total_time_ms = (time.perf_counter() - t_total_start) * 1000.0
        timings["total_ms"] = total_time_ms

        metadata: Dict[str, Any] = {
            "pipeline_type": "enhanced",
            "timings_ms": timings,
            "intermediates": intermediates,
            "skin_quality": skin_meta.get("quality", {}),
            "feature_info": feat_meta,
            "feature_names": feature_names,
            "feature_dim": len(features),
        }
        return features, metadata


# ==============================================================================
# 3. HÀM XỬ LÝ ẢNH ĐƠN LẺ VÀ SO SÁNH (SINGLE IMAGE & COMPARISON)
# ==============================================================================

def process_single_image(
    image_input: Union[str, Path, np.ndarray],
    pipeline_type: str = "enhanced",
    color_space: str = "RGB",
    feature_group: str = "all",
    **pipeline_kwargs: Any
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Xử lý một ảnh đơn lẻ phục vụ kiểm thử nhanh hoặc tích hợp giao diện Streamlit/Web API.
    Process a single image using specified pipeline type.

    Args:
        image_input: Đường dẫn tệp ảnh hoặc mảng np.ndarray.
        pipeline_type: 'baseline' hoặc 'enhanced'.
        color_space: 'RGB', 'HSV', hoặc 'YCbCr'.
        feature_group: 'color', 'brightness', 'texture', hoặc 'all'.
        pipeline_kwargs: Các tham số bổ sung cho pipeline.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: Vector đặc trưng và metadata.
    """
    pt = pipeline_type.lower()
    if pt == "baseline":
        pipe = BaselinePipeline(
            color_space=color_space, feature_group=feature_group, **pipeline_kwargs
        )
    else:
        pipe = EnhancedPipeline(
            color_space=color_space, feature_group=feature_group, **pipeline_kwargs
        )

    return pipe.process(image_input)


def compare_pipelines_on_image(
    image_input: Union[str, Path, np.ndarray],
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    So sánh trực quan và định lượng giữa Baseline và Enhanced Pipeline trên cùng 1 ảnh.
    Compares baseline and enhanced pipelines side-by-side on a single image.

    Xuất biểu đồ 2 hàng x 4 cột:
    - Hàng 1 (Baseline): Ảnh gốc -> Ảnh Resize -> Mặt nạ thô -> Vùng da trích xuất
    - Hàng 2 (Enhanced): Khử nhiễu -> Tăng tương phản CLAHE -> Mặt nạ làm sạch -> Vùng da trích xuất

    Args:
        image_input: Đường dẫn ảnh hoặc mảng ảnh.
        save_path: Đường dẫn lưu biểu đồ kết quả (tùy chọn).

    Returns:
        Dict[str, Any]: Bảng so sánh chỉ số thời gian và chất lượng phân đoạn da.
    """
    base_pipe = BaselinePipeline()
    enh_pipe = EnhancedPipeline()

    feat_base, meta_base = base_pipe.process(image_input)
    feat_enh, meta_enh = enh_pipe.process(image_input)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "So Sánh Đường Ống Xử Lý: Baseline vs Enhanced Pipeline",
        fontsize=14,
        fontweight="bold"
    )

    inter_base = meta_base["intermediates"]
    inter_enh = meta_enh["intermediates"]

    # HÀNG 1: BASELINE
    axes[0, 0].imshow(inter_base["raw_rgb"])
    axes[0, 0].set_title("1. Baseline: Ảnh Gốc", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(inter_base["resized_rgb"])
    axes[0, 1].set_title("2. Baseline: Resize Trực Tiếp", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(inter_base["skin_mask"], cmap="gray")
    axes[0, 2].set_title(
        f"3. Baseline: Mặt Nạ Thô HSV\n(Skin: {meta_base['skin_quality'].get('skin_percentage', 0):.1f}%)",
        fontsize=10
    )
    axes[0, 2].axis("off")

    axes[0, 3].imshow(inter_base["extracted_skin_rgb"])
    axes[0, 3].set_title(
        f"4. Baseline: Vùng Da Tách\n(Thời gian: {meta_base['timings_ms']['total_ms']:.1f} ms)",
        fontsize=10,
        fontweight="bold"
    )
    axes[0, 3].axis("off")

    # HÀNG 2: ENHANCED
    axes[1, 0].imshow(inter_enh["denoised_rgb"])
    axes[1, 0].set_title("1. Enhanced: Lọc Nhiễu Gaussian", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(inter_enh["enhanced_rgb"])
    axes[1, 1].set_title("2. Enhanced: Tăng Tương Phản CLAHE", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(inter_enh["cleaned_mask"], cmap="gray")
    axes[1, 2].set_title(
        f"3. Enhanced: Mặt Nạ Hình Thái Học\n(Skin: {meta_enh['skin_quality'].get('skin_percentage', 0):.1f}%)",
        fontsize=10
    )
    axes[1, 2].axis("off")

    axes[1, 3].imshow(inter_enh["extracted_skin_rgb"])
    axes[1, 3].set_title(
        f"4. Enhanced: Vùng Da Tách\n(Thời gian: {meta_enh['timings_ms']['total_ms']:.1f} ms)",
        fontsize=10,
        fontweight="bold"
    )
    axes[1, 3].axis("off")

    plt.tight_layout()

    if save_path:
        out_f = Path(save_path)
        out_f.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_f, dpi=200, bbox_inches="tight")
        plt.close(fig)

    comparison_result = {
        "baseline_timing_ms": meta_base["timings_ms"],
        "enhanced_timing_ms": meta_enh["timings_ms"],
        "baseline_skin_quality": meta_base["skin_quality"],
        "enhanced_skin_quality": meta_enh["skin_quality"],
        "feature_dim": len(feat_base),
        "differences": {
            "timing_diff_ms": round(
                meta_enh["timings_ms"]["total_ms"] - meta_base["timings_ms"]["total_ms"], 2
            ),
            "skin_ratio_diff": round(
                meta_enh["skin_quality"].get("skin_ratio", 0) - meta_base["skin_quality"].get("skin_ratio", 0), 4
            )
        }
    }
    return comparison_result


# ==============================================================================
# 4. XỬ LÝ TOÀN BỘ TẬP DỮ LIỆU THEO LÔ (BATCH DATASET PROCESSING)
# ==============================================================================

def process_dataset(
    image_paths: List[Union[str, Path]],
    labels: Optional[List[Union[str, int]]] = None,
    pipeline_type: str = "enhanced",
    color_space: str = "RGB",
    feature_group: str = "all",
    show_progress: bool = True,
    **pipeline_kwargs: Any
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """
    Xử lý hàng loạt toàn bộ danh sách tệp ảnh trong bộ dữ liệu qua pipeline.
    Batch process dataset of image files through specified pipeline.

    Args:
        image_paths: Danh sách đường dẫn tới các tệp ảnh.
        labels: Danh sách nhãn tương ứng (tùy chọn).
        pipeline_type: 'baseline' hoặc 'enhanced'.
        color_space: 'RGB', 'HSV', hoặc 'YCbCr'.
        feature_group: Cấu hình nhóm đặc trưng trích xuất.
        show_progress: Hiển thị thanh tiến trình tqdm.
        pipeline_kwargs: Cấu hình bổ sung cho pipeline.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
            - X: Ma trận đặc trưng [N_samples, N_features] kiểu np.float32.
            - y: Mảng nhãn 1D [N_samples] (nếu có cung cấp).
            - metadata_df: Bảng DataFrame ghi nhận chi tiết từng ảnh: đường dẫn,
              nhãn, thời gian xử lý, tỷ lệ điểm ảnh da, tính hợp lệ.
    """
    if not image_paths:
        raise ValueError("[Lỗi] Danh sách ảnh rỗng.")

    pt = pipeline_type.lower()
    if pt == "baseline":
        pipeline = BaselinePipeline(
            color_space=color_space, feature_group=feature_group, **pipeline_kwargs
        )
    else:
        pipeline = EnhancedPipeline(
            color_space=color_space, feature_group=feature_group, **pipeline_kwargs
        )

    feature_matrix_list: List[np.ndarray] = []
    valid_labels_list: List[Any] = []
    records: List[Dict[str, Any]] = []

    iterator = tqdm(
        enumerate(image_paths),
        total=len(image_paths),
        desc=f"Xử lý {pt.capitalize()} Pipeline",
        disable=not show_progress
    )

    for idx, img_path in iterator:
        p_obj = Path(img_path)
        lbl = labels[idx] if (labels is not None and idx < len(labels)) else None

        try:
            feats, meta = pipeline.process(p_obj)
            feature_matrix_list.append(feats)
            if lbl is not None:
                valid_labels_list.append(lbl)

            q = meta.get("skin_quality", {})
            t = meta.get("timings_ms", {})
            records.append({
                "image_path": str(p_obj),
                "filename": p_obj.name,
                "label": lbl,
                "skin_percentage": q.get("skin_percentage", 0.0),
                "skin_pixels": q.get("skin_pixels", 0),
                "is_valid_mask": q.get("is_valid", False),
                "total_time_ms": t.get("total_ms", 0.0),
                "status": "success",
                "error": None
            })

        except Exception as exc:
            logger.warning(f"Lỗi khi xử lý ảnh {p_obj.name}: {exc}")
            # Lấy vector mẫu số 0 nếu ảnh lỗi để không làm đứt chuỗi dữ liệu
            expected_dim = len(get_feature_names(feature_group=feature_group, color_space=color_space))
            feature_matrix_list.append(np.zeros(expected_dim, dtype=np.float32))
            if lbl is not None:
                valid_labels_list.append(lbl)

            records.append({
                "image_path": str(p_obj),
                "filename": p_obj.name,
                "label": lbl,
                "skin_percentage": 0.0,
                "skin_pixels": 0,
                "is_valid_mask": False,
                "total_time_ms": 0.0,
                "status": "failed",
                "error": str(exc)
            })

    X = np.vstack(feature_matrix_list).astype(np.float32)
    y = np.array(valid_labels_list) if labels is not None else None
    metadata_df = pd.DataFrame(records)

    logger.info(
        f"Hoàn thành xử lý lô: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng. "
        f"Thành công: {sum(metadata_df['status'] == 'success')}/{len(metadata_df)}"
    )

    return X, y, metadata_df
