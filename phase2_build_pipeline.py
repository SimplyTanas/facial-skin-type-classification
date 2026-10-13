"""
Kịch bản Thực thi Giai đoạn 2: Xây Dựng Đường Ống Xử Lý Ảnh và Trích Xuất Đặc Trưng
(Phase 2: Image Processing Pipeline & Feature Engineering Workflow)
==================================================================================
Đề tài: Phân loại loại da mặt (Dry, Normal, Oily) sử dụng Kỹ thuật Xử lý ảnh số & Học máy.

Mục tiêu quy trình Phase 2:
    - Bước 1: Tải các mẫu ảnh đại diện cho từng loại da (Dry, Normal, Oily).
    - Bước 2: Thử nghiệm tiền xử lý cơ bản (Baseline Preprocessing: Resize trực tiếp).
    - Bước 3: Thử nghiệm tiền xử lý nâng cao (Enhanced Preprocessing: Gaussian Blur + CLAHE).
    - Bước 4: Thử nghiệm và so sánh phân vùng da (Skin Detection: HSV vs YCrCb vs Phối hợp đa không gian).
    - Bước 5: So sánh biểu diễn qua các không gian màu (RGB, HSV, YCrCb, LAB, Grayscale).
    - Bước 6: Trích xuất toàn diện 3 nhóm đặc trưng (Nhóm A: Màu, Nhóm B: Độ sáng, Nhóm C: LBP).
    - Bước 7: Xử lý lô toàn bộ tập dữ liệu qua cả BaselinePipeline và EnhancedPipeline.
    - Bước 8: Chuẩn hóa Z-Score, giảm chiều PCA và lưu trữ ma trận đặc trưng (.npz, .joblib).
    - Bước 9: Đánh giá so sánh định lượng (Latency thời gian, chất lượng mặt nạ, đặc trưng trích xuất).

Cách sử dụng:
    python phase2_build_pipeline.py
    python phase2_build_pipeline.py --data-dir "path/to/dataset"
    python phase2_build_pipeline.py --max-samples 30  # Giới hạn số mẫu xử lý demo nhanh
    python phase2_build_pipeline.py --use-pca         # Bật giảm chiều PCA cho đặc trưng
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Cấu hình UTF-8 cho console Windows tránh UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Thiết lập đường dẫn import module nội bộ
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.color_space import (
    compare_color_spaces,
    convert_to_gray,
    convert_to_hsv,
    convert_to_ycrcb,
    split_channels,
    visualize_channels,
)
from src.data_loader import discover_dataset_structure, load_dataset_paths
from src.feature_extraction import (
    extract_brightness_features,
    extract_features,
    extract_group_a_color_features,
    extract_lbp_features,
    get_feature_names,
)
from src.feature_normalization import FeatureNormalizer
from src.pipeline import (
    BaselinePipeline,
    EnhancedPipeline,
    compare_pipelines_on_image,
    process_dataset,
    process_single_image,
)
from src.preprocessing import (
    apply_clahe,
    apply_gaussian_blur,
    bgr_to_rgb,
    read_image,
    resize_image,
)
from src.skin_detection import (
    clean_skin_mask,
    create_skin_mask,
    detect_skin_hsv,
    detect_skin_ycrcb,
    extract_roi,
    visualize_skin_detection,
)
from src.utils import (
    Timer,
    ensure_output_dirs,
    format_duration,
    plot_image_grid,
    save_json,
    setup_logger,
)

logger = setup_logger("phase2_pipeline")


# ==============================================================================
# HÀM BỔ TRỢ: NẠP HOẶC TẠO ẢNH MẪU ĐẠI DIỆN
# ==============================================================================

def get_representative_samples(data_dir: Path) -> Dict[str, Path]:
    """
    Tìm hoặc tạo 1 ảnh mẫu đại diện cho mỗi lớp (dry, normal, oily) để chạy demo.
    """
    samples: Dict[str, Path] = {}
    class_names = list(getattr(config, "CLASS_NAMES", ["dry", "normal", "oily"]))

    # 1. Tìm trong thư mục train hoặc cấu trúc phẳng của data_dir
    for c in class_names:
        candidate_paths = [
            data_dir / "train" / c,
            data_dir / c,
            data_dir / "valid" / c,
            data_dir / "test" / c,
        ]
        found_file = None
        for cand in candidate_paths:
            if cand.exists() and cand.is_dir():
                imgs = [p for p in cand.glob("*.*") if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS]
                if imgs:
                    found_file = imgs[0]
                    break
        if found_file:
            samples[c] = found_file

    # 2. Nếu thiếu lớp nào, tự động gọi bộ tạo dữ liệu mẫu mô phỏng từ phase1
    if len(samples) < len(class_names):
        logger.info("Chưa tìm thấy đủ dữ liệu trên đĩa. Đang tự động tạo dữ liệu mẫu kiểm thử...")
        try:
            from phase1_data_preparation import create_synthetic_sample_dataset
            create_synthetic_sample_dataset(data_dir, images_per_class=6)
        except Exception as e:
            logger.warning(f"Không thể import synthetic generator từ phase 1: {e}. Tạo ảnh mẫu trực tiếp.")
            data_dir.mkdir(parents=True, exist_ok=True)
            for c in class_names:
                c_dir = data_dir / "train" / c
                c_dir.mkdir(parents=True, exist_ok=True)
                sample_file = c_dir / f"sample_{c}.jpg"
                dummy_img = np.full((250, 250, 3), (150, 180, 220), dtype=np.uint8)
                cv2.imwrite(str(sample_file), dummy_img)

        # Quét lại
        for c in class_names:
            c_dir = data_dir / "train" / c
            if c_dir.exists():
                imgs = [p for p in c_dir.glob("*.*") if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS]
                if imgs:
                    samples[c] = imgs[0]

    return samples


# ==============================================================================
# QUY TRÌNH CHÍNH (MAIN WORKFLOW)
# ==============================================================================

def run_phase2_workflow(
    data_dir: Optional[Path] = None,
    max_samples: Optional[int] = 30,
    color_space: str = "RGB",
    feature_group: str = "all",
    use_pca: bool = True,
    pca_components: float = 0.95
) -> None:
    """
    Thực thi tuần tự đầy đủ 9 bước của Phase 2.
    """
    print("\n" + "=" * 80)
    print(" BẮT ĐẦU QUY TRÌNH PHASE 2: IMAGE PROCESSING PIPELINE & FEATURE ENGINEERING")
    print(" Đề tài: Phân loại loại da mặt (Dry, Normal, Oily) sử dụng Kỹ thuật Xử lý ảnh số")
    print("=" * 80)

    # Đảm bảo cây thư mục xuất kết quả
    dirs = ensure_output_dirs()
    figures_dir = dirs["figures"]
    processed_dir = dirs["processed_data"]
    models_dir = dirs["models"]
    reports_dir = dirs["reports"]

    target_data_dir = data_dir if data_dir is not None else config.DATA_DIR
    logger.info(f"Thư mục dữ liệu: {target_data_dir}")

    # --------------------------------------------------------------------------
    # BƯỚC 1: TẢI CÁC ẢNH MẪU ĐẠI DIỆN CHO TỪNG LOẠI DA
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> BƯỚC 1: Tải các ảnh mẫu đại diện từng loại da (Dry, Normal, Oily)...")
    print("-" * 60)

    sample_images = get_representative_samples(target_data_dir)
    for c_name, p in sample_images.items():
        print(f"  [+] Lớp {c_name.upper():<7}: {p.name} (Đường dẫn: {p})")

    # Chọn 1 ảnh mẫu chính (ưu tiên ảnh oily hoặc normal để kiểm tra độ chói dầu nhờn)
    demo_class = "oily" if "oily" in sample_images else list(sample_images.keys())[0]
    demo_image_path = sample_images[demo_class]
    demo_raw_bgr = read_image(demo_image_path, as_rgb=False)
    demo_raw_rgb = bgr_to_rgb(demo_raw_bgr)

    # --------------------------------------------------------------------------
    # BƯỚC 2: DEMO TIỀN XỬ LÝ CƠ BẢN (BASELINE PREPROCESSING)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> BƯỚC 2: Thử nghiệm tiền xử lý cơ bản (Baseline Preprocessing)...")
    print("-" * 60)

    with Timer("Baseline Preprocessing") as t_base_pre:
        base_resized_bgr, _ = resize_image(demo_raw_bgr, target_size=(224, 224), keep_aspect_ratio=False)
    print(f"  [i] Kích thước gốc   : {demo_raw_bgr.shape[1]}x{demo_raw_bgr.shape[0]} px")
    print(f"  [i] Kích thước sau resize: {base_resized_bgr.shape[1]}x{base_resized_bgr.shape[0]} px")
    print(f"  [i] Thời gian thực thi: {t_base_pre.formatted}")

    fig_step2, ax2 = plt.subplots(1, 2, figsize=(8, 4))
    ax2[0].imshow(demo_raw_rgb)
    ax2[0].set_title(f"Ảnh Gốc ({demo_raw_bgr.shape[1]}x{demo_raw_bgr.shape[0]})", fontweight="bold")
    ax2[0].axis("off")
    ax2[1].imshow(bgr_to_rgb(base_resized_bgr))
    ax2[1].set_title("Baseline Resize (224x224)", fontweight="bold")
    ax2[1].axis("off")
    plt.tight_layout()
    fig_step2.savefig(figures_dir / "phase2_step2_baseline_preprocessing.png", dpi=200)
    plt.close(fig_step2)
    print(f"  [✓] Đã lưu biểu đồ: {figures_dir / 'phase2_step2_baseline_preprocessing.png'}")

    # --------------------------------------------------------------------------
    # BƯỚC 3: DEMO TIỀN XỬ LÝ NÂNG CAO (ENHANCED PREPROCESSING)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> BƯỚC 3: Thử nghiệm tiền xử lý nâng cao (Gaussian Denoise + CLAHE)...")
    print("-" * 60)

    with Timer("Enhanced Preprocessing") as t_enh_pre:
        enh_denoised_bgr, _ = apply_gaussian_blur(base_resized_bgr, kernel_size=(5, 5), sigma=1.0)
        enh_clahe_bgr, _ = apply_clahe(enh_denoised_bgr, clip_limit=2.0, tile_grid_size=(8, 8), color_space="LAB", is_bgr=True)
    print(f"  [i] Khử nhiễu làm mịn bề mặt: Gaussian Kernel (5, 5), Sigma=1.0")
    print(f"  [i] Cân bằng tương phản thích ứng: CLAHE trên kênh Luminance L (Clip=2.0, Grid=8x8)")
    print(f"  [i] Thời gian thực thi: {t_enh_pre.formatted}")

    fig_step3, ax3 = plt.subplots(1, 3, figsize=(12, 4))
    ax3[0].imshow(bgr_to_rgb(base_resized_bgr))
    ax3[0].set_title("1. Ảnh Sau Resize", fontweight="bold")
    ax3[0].axis("off")
    ax3[1].imshow(bgr_to_rgb(enh_denoised_bgr))
    ax3[1].set_title("2. Khử Nhiễu Gaussian", fontweight="bold")
    ax3[1].axis("off")
    ax3[2].imshow(bgr_to_rgb(enh_clahe_bgr))
    ax3[2].set_title("3. Tăng Tương Phản CLAHE", fontweight="bold")
    ax3[2].axis("off")
    plt.tight_layout()
    fig_step3.savefig(figures_dir / "phase2_step3_enhanced_preprocessing.png", dpi=200)
    plt.close(fig_step3)
    print(f"  [✓] Đã lưu biểu đồ: {figures_dir / 'phase2_step3_enhanced_preprocessing.png'}")

    # --------------------------------------------------------------------------
    # BƯỚC 4: DEMO PHÂN VÙNG DA (SKIN DETECTION: HSV vs YCrCb vs COMBINED)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> BƯỚC 4: Thử nghiệm và so sánh các phương pháp phân đoạn da...")
    print("-" * 60)

    # 1. Ngưỡng HSV
    mask_hsv, _ = detect_skin_hsv(enh_clahe_bgr, is_bgr=True)
    # 2. Ngưỡng YCrCb
    mask_ycrcb, _ = detect_skin_ycrcb(enh_clahe_bgr, is_bgr=True)
    # 3. Phối hợp (AND)
    mask_combined, _ = create_skin_mask(enh_clahe_bgr, method="combined", is_bgr=True)
    # 4. Làm sạch hình thái học (Morphology Opening + Closing + Hole Filling)
    cleaned_mask, clean_meta = clean_skin_mask(mask_combined, min_region_area=150, fill_holes=True)
    extracted_skin_bgr, _ = extract_roi(enh_clahe_bgr, cleaned_mask)

    print(f"  [i] Tỷ lệ diện tích da HSV      : {np.count_nonzero(mask_hsv > 0) / mask_hsv.size * 100:.2f}%")
    print(f"  [i] Tỷ lệ diện tích da YCrCb    : {np.count_nonzero(mask_ycrcb > 0) / mask_ycrcb.size * 100:.2f}%")
    print(f"  [i] Tỷ lệ sau phối hợp (AND)   : {np.count_nonzero(mask_combined > 0) / mask_combined.size * 100:.2f}%")
    print(f"  [i] Tỷ lệ sau Morphology sạch   : {np.count_nonzero(cleaned_mask > 0) / cleaned_mask.size * 100:.2f}%")

    fig_step4, ax4 = plt.subplots(1, 5, figsize=(18, 4))
    ax4[0].imshow(mask_hsv, cmap="gray")
    ax4[0].set_title("1. Mặt nạ HSV", fontweight="bold")
    ax4[0].axis("off")
    ax4[1].imshow(mask_ycrcb, cmap="gray")
    ax4[1].set_title("2. Mặt nạ YCrCb", fontweight="bold")
    ax4[1].axis("off")
    ax4[2].imshow(mask_combined, cmap="gray")
    ax4[2].set_title("3. Phối hợp (AND)", fontweight="bold")
    ax4[2].axis("off")
    ax4[3].imshow(cleaned_mask, cmap="gray")
    ax4[3].set_title("4. Morphology Clean", fontweight="bold")
    ax4[3].axis("off")
    ax4[4].imshow(bgr_to_rgb(extracted_skin_bgr))
    ax4[4].set_title("5. Vùng Da Trích Xuất", fontweight="bold")
    ax4[4].axis("off")
    plt.tight_layout()
    fig_step4.savefig(figures_dir / "phase2_step4_skin_detection_comparison.png", dpi=200)
    plt.close(fig_step4)
    print(f"  [✓] Đã lưu biểu đồ: {figures_dir / 'phase2_step4_skin_detection_comparison.png'}")

    # --------------------------------------------------------------------------
    # BƯỚC 5: SO SÁNH BIỂU DIỄN QUA CÁC KHÔNG GIAN MÀU
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> BƯỚC 5: So sánh biểu diễn qua các không gian màu...")
    print("-" * 60)

    fig_step5 = compare_color_spaces(demo_raw_bgr, is_bgr=True)
    fig_step5.savefig(figures_dir / "phase2_step5_color_spaces.png", dpi=200, bbox_inches="tight")
    plt.close(fig_step5)
    print(f"  [✓] Đã lưu biểu đồ: {figures_dir / 'phase2_step5_color_spaces.png'}")

    # --------------------------------------------------------------------------
    # BƯỚC 6: DEMO TRÍCH XUẤT ĐẶC TRƯNG TẤT CẢ CÁC NHÓM (A, B, C)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> BƯỚC 6: Thử nghiệm trích xuất các nhóm đặc trưng (Nhóm A, B, C)...")
    print("-" * 60)

    feat_vector, feat_names, feat_info = extract_features(
        image=enh_clahe_bgr,
        mask=cleaned_mask,
        feature_group="all",
        color_space=color_space,
        hist_bins=32,
        k_dominant=1,
        lbp_radius=1,
        lbp_n_points=8,
        is_bgr=True
    )

    print(f"  [+] Tổng số chiều đặc trưng trích xuất : {len(feat_vector)} chiều")
    print(f"      - Nhóm A (Color Moments + Hist + Dom): {feat_info['subgroups'].get('color_dim', 0)} chiều")
    print(f"      - Nhóm B (Brightness & Highlights)   : {feat_info['subgroups'].get('brightness_dim', 0)} chiều")
    print(f"      - Nhóm C (Texture LBP Hist & Stats)  : {feat_info['subgroups'].get('texture_dim', 0)} chiều")

    # Vẽ biểu đồ trực quan hóa vector đặc trưng
    fig_step6, ax6 = plt.subplots(3, 1, figsize=(14, 8))
    fig_step6.suptitle("Phân Tích Chi Tiết 3 Nhóm Đặc Trưng Trên Vùng Da Mặt", fontsize=14, fontweight="bold")

    dim_a = feat_info['subgroups'].get('color_dim', 105)
    dim_b = feat_info['subgroups'].get('brightness_dim', 7)
    dim_c = feat_info['subgroups'].get('texture_dim', 14)

    # 1. Nhóm A
    ax6[0].plot(feat_vector[:dim_a], color="#E69F00", lw=1.5)
    ax6[0].set_title(f"Nhóm A: Đặc Trưng Màu Sắc ({dim_a} chiều: Mean, Std, Histogram 32 bins, Dominant Color)", fontsize=10, fontweight="bold")
    ax6[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Nhóm B
    ax6[1].bar(range(dim_b), feat_vector[dim_a:dim_a + dim_b], color="#0072B2")
    b_names = feat_names[dim_a:dim_a + dim_b]
    ax6[1].set_xticks(range(dim_b))
    ax6[1].set_xticklabels([n.split("_")[-1] for n in b_names], rotation=15, fontsize=8)
    ax6[1].set_title(f"Nhóm B: Đặc Trưng Độ Sáng ({dim_b} chiều: Mean, Std, P25, Median, P75, Range, Highlight Ratio)", fontsize=10, fontweight="bold")
    ax6[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Nhóm C
    ax6[2].plot(feat_vector[dim_a + dim_b:], color="#009E73", lw=1.5, marker="o", markersize=3)
    ax6[2].set_title(f"Nhóm C: Đặc Trưng Kết Cấu LBP ({dim_c} chiều: Uniform LBP Histogram + Stats)", fontsize=10, fontweight="bold")
    ax6[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_step6.savefig(figures_dir / "phase2_step6_features_breakdown.png", dpi=200)
    plt.close(fig_step6)
    print(f"  [✓] Đã lưu biểu đồ: {figures_dir / 'phase2_step6_features_breakdown.png'}")

    # --------------------------------------------------------------------------
    # BƯỚC 7: XỬ LÝ TOÀN BỘ TẬP DỮ LIỆU QUA CẢ HAI PIPELINE (BATCH PROCESSING)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> BƯỚC 7: Xử lý lô bộ dữ liệu qua BaselinePipeline và EnhancedPipeline...")
    print("-" * 60)

    # Thu thập danh sách ảnh
    image_paths_dict = load_dataset_paths(target_data_dir)
    all_paths: List[Path] = []
    all_labels: List[str] = []

    for split_key in ["train", "valid", "test", "all"]:
        if split_key in image_paths_dict:
            for p, lbl in image_paths_dict[split_key]:
                all_paths.append(p)
                all_labels.append(lbl)

    # Nếu số lượng ảnh lớn hơn max_samples, lấy mẫu đại diện để chạy nhanh
    if max_samples and len(all_paths) > max_samples:
        print(f"  [i] Giới hạn xử lý {max_samples}/{len(all_paths)} ảnh để chạy thử nghiệm nhanh.")
        rng = np.random.RandomState(42)
        idx_sub = rng.choice(len(all_paths), size=max_samples, replace=False)
        all_paths = [all_paths[i] for i in idx_sub]
        all_labels = [all_labels[i] for i in idx_sub]

    print(f"  [+] Đang xử lý {len(all_paths)} ảnh qua BaselinePipeline...")
    X_base, y_base, df_base = process_dataset(
        image_paths=all_paths,
        labels=all_labels,
        pipeline_type="baseline",
        color_space=color_space,
        feature_group=feature_group,
        show_progress=True
    )

    print(f"  [+] Đang xử lý {len(all_paths)} ảnh qua EnhancedPipeline...")
    X_enh, y_enh, df_enh = process_dataset(
        image_paths=all_paths,
        labels=all_labels,
        pipeline_type="enhanced",
        color_space=color_space,
        feature_group=feature_group,
        show_progress=True
    )

    # --------------------------------------------------------------------------
    # BƯỚC 8: CHUẨN HÓA VÀ LƯU TRỮ MA TRẬN ĐẶC TRƯNG
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> BƯỚC 8: Chuẩn hóa đặc trưng Z-Score, giảm chiều PCA và lưu trữ...")
    print("-" * 60)

    # Khởi tạo và fit FeatureNormalizer CHỈ trên ma trận train
    normalizer = FeatureNormalizer(
        use_pca=use_pca,
        n_components=pca_components,
        random_state=42
    )
    X_enh_normalized = normalizer.fit_transform(X_enh, feature_names=feat_names)

    norm_summary = normalizer.summary()
    print(f"  [+] Số chiều gốc        : {norm_summary['n_features_in']} chiều")
    print(f"  [+] Số chiều sau PCA    : {norm_summary['n_features_out']} chiều (Phương sai: {norm_summary.get('pca_cumulative_variance', 1.0) * 100:.2f}%)")

    # Lưu trữ các file đặc trưng
    np.savez_compressed(
        processed_dir / "features_baseline.npz",
        X=X_base,
        y=y_base,
        feature_names=feat_names
    )
    np.savez_compressed(
        processed_dir / "features_enhanced.npz",
        X=X_enh,
        X_normalized=X_enh_normalized,
        y=y_enh,
        feature_names=feat_names
    )
    df_enh.to_csv(processed_dir / "dataset_processing_metadata.csv", index=False)
    normalizer.save(models_dir / "feature_normalizer.joblib")

    print(f"  [✓] Đã lưu ma trận đặc trưng Baseline : {processed_dir / 'features_baseline.npz'}")
    print(f"  [✓] Đã lưu ma trận đặc trưng Enhanced : {processed_dir / 'features_enhanced.npz'}")
    print(f"  [✓] Đã lưu bộ chuẩn hóa Normalizer    : {models_dir / 'feature_normalizer.joblib'}")

    # --------------------------------------------------------------------------
    # BƯỚC 9: SO SÁNH ĐỊNH LƯỢNG BASELINE VS ENHANCED
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> BƯỚC 9: So sánh định lượng kết quả giữa Baseline và Enhanced...")
    print("-" * 60)

    avg_time_base = float(df_base["total_time_ms"].mean())
    avg_time_enh = float(df_enh["total_time_ms"].mean())
    avg_skin_base = float(df_base["skin_percentage"].mean())
    avg_skin_enh = float(df_enh["skin_percentage"].mean())
    valid_base_ratio = float((df_base["is_valid_mask"].sum() / len(df_base)) * 100)
    valid_enh_ratio = float((df_enh["is_valid_mask"].sum() / len(df_enh)) * 100)

    print(f"  [Thống kê so sánh]")
    print(f"  1. Thời gian xử lý trung bình/ảnh: Baseline = {avg_time_base:.1f} ms | Enhanced = {avg_time_enh:.1f} ms")
    print(f"  2. Tỷ lệ diện tích da trung bình  : Baseline = {avg_skin_base:.2f}% | Enhanced = {avg_skin_enh:.2f}%")
    print(f"  3. Tỷ lệ mặt nạ đạt chuẩn (>5% da): Baseline = {valid_base_ratio:.1f}% | Enhanced = {valid_enh_ratio:.1f}%")

    # Vẽ biểu đồ so sánh định lượng
    fig_step9, ax9 = plt.subplots(1, 3, figsize=(15, 4.5))
    fig_step9.suptitle("So Sánh Hiệu Năng: Baseline vs Enhanced Pipeline", fontsize=13, fontweight="bold")

    labels_bar = ["Baseline", "Enhanced"]
    colors = ["#E69F00", "#0072B2"]

    # 1. Thời gian
    ax9[0].bar(labels_bar, [avg_time_base, avg_time_enh], color=colors, width=0.5)
    ax9[0].set_ylabel("Thời gian (ms)")
    ax9[0].set_title("Thời Gian Xử Lý Trung Bình", fontweight="bold")
    for i, v in enumerate([avg_time_base, avg_time_enh]):
        ax9[0].text(i, v + max(2, v * 0.03), f"{v:.1f} ms", ha="center", fontweight="bold")
    ax9[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Tỷ lệ da
    ax9[1].bar(labels_bar, [avg_skin_base, avg_skin_enh], color=colors, width=0.5)
    ax9[1].set_ylabel("Phần trăm da (%)")
    ax9[1].set_title("Độ Bao Phủ Vùng Da Mặt", fontweight="bold")
    for i, v in enumerate([avg_skin_base, avg_skin_enh]):
        ax9[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax9[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Tỷ lệ mặt nạ hợp lệ
    ax9[2].bar(labels_bar, [valid_base_ratio, valid_enh_ratio], color=colors, width=0.5)
    ax9[2].set_ylabel("Tỷ lệ hợp lệ (%)")
    ax9[2].set_title("Tỷ Lệ Mặt Nạ Đạt Chuẩn", fontweight="bold")
    for i, v in enumerate([valid_base_ratio, valid_enh_ratio]):
        ax9[2].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax9[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_step9.savefig(figures_dir / "phase2_step9_pipeline_comparison.png", dpi=200)
    plt.close(fig_step9)
    print(f"  [✓] Đã lưu biểu đồ: {figures_dir / 'phase2_step9_pipeline_comparison.png'}")

    # Chạy thêm so sánh chi tiết trên 1 ảnh mẫu
    compare_pipelines_on_image(demo_image_path, save_path=figures_dir / "phase2_single_image_comparison.png")

    # Lưu báo cáo tổng kết JSON
    summary_report = {
        "dataset_samples_processed": len(all_paths),
        "feature_dimension_raw": len(feat_vector),
        "feature_dimension_pca": norm_summary["n_features_out"],
        "baseline_metrics": {
            "avg_latency_ms": round(avg_time_base, 2),
            "avg_skin_coverage_pct": round(avg_skin_base, 2),
            "valid_mask_pct": round(valid_base_ratio, 2),
        },
        "enhanced_metrics": {
            "avg_latency_ms": round(avg_time_enh, 2),
            "avg_skin_coverage_pct": round(avg_skin_enh, 2),
            "valid_mask_pct": round(valid_enh_ratio, 2),
        },
        "pca_explained_variance_pct": round(norm_summary.get("pca_cumulative_variance", 1.0) * 100, 2),
    }
    save_json(summary_report, reports_dir / "phase2_pipeline_summary.json")

    print("\n" + "=" * 80)
    print(" TỔNG KẾT GIAI ĐOẠN PHASE 2 HOÀN TẤT THÀNH CÔNG!")
    print(f"  - Các biểu đồ trực quan hóa đã xuất tại: {figures_dir}")
    print(f"  - Ma trận đặc trưng đã xuất tại       : {processed_dir}")
    print(f"  - Mô hình bộ chuẩn hóa đã lưu tại     : {models_dir}")
    print(f"  - Báo cáo định lượng JSON lưu tại      : {reports_dir / 'phase2_pipeline_summary.json'}")
    print("=" * 80 + "\n")


# ==============================================================================
# ĐIỂM VÀO DÒNG LỆNH (COMMAND LINE INTERFACE)
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Image Processing Pipeline & Feature Engineering Workflow"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Đường dẫn tới thư mục bộ dữ liệu (mặc định lấy config.DATA_DIR)."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Số lượng mẫu tối đa để xử lý batch trong bước demo (mặc định 30 mẫu)."
    )
    parser.add_argument(
        "--color-space",
        type=str,
        default="RGB",
        choices=["RGB", "HSV", "YCbCr"],
        help="Không gian màu sử dụng cho trích xuất đặc trưng (mặc định RGB)."
    )
    parser.add_argument(
        "--feature-group",
        type=str,
        default="all",
        choices=["color", "brightness", "texture", "color_brightness", "all"],
        help="Nhóm đặc trưng cần trích xuất (mặc định all)."
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        default=True,
        help="Áp dụng giảm chiều PCA cho ma trận đặc trưng (mặc định bật)."
    )
    parser.add_argument(
        "--pca-components",
        type=float,
        default=0.95,
        help="Tỷ lệ phương sai giải thích của PCA (mặc định 0.95)."
    )

    args = parser.parse_args()

    data_dir_path = Path(args.data_dir).resolve() if args.data_dir else None

    run_phase2_workflow(
        data_dir=data_dir_path,
        max_samples=args.max_samples,
        color_space=args.color_space,
        feature_group=args.feature_group,
        use_pca=args.use_pca,
        pca_components=args.pca_components
    )


if __name__ == "__main__":
    main()
