"""
Kịch bản Thực thi Giai đoạn 1: Chuẩn bị Dữ liệu (Phase 1: Data Preparation Pipeline)
==================================================================================
Đề tài: Phân loại loại da mặt (Dry, Normal, Oily) sử dụng Kỹ thuật Xử lý ảnh số & Học máy.

Mục tiêu quy trình Phase 1:
    - Bước 1: Khám phá và nhận diện cấu trúc cây thư mục bộ dữ liệu (Dataset Discovery).
    - Bước 2: Tải và thẩm định tính toàn vẹn từng tệp ảnh (Validation).
    - Bước 3: Tính toán các chỉ số thống kê bộ dữ liệu (Dimensions, Aspect Ratio, File Sizes).
    - Bước 4: Kiểm tra và phát hiện ảnh trùng lặp tuyệt đối (MD5) và cảm nhận (dHash).
    - Bước 5: Phân tích chất lượng ảnh số (Độ mờ Laplacian, độ sáng quá mức/quá tối, độ tương phản).
    - Bước 6: Kiểm tra tính cân bằng và tỉ lệ phân chia phân tập (Train / Valid / Test Split).
    - Bước 7: Trực quan hóa dữ liệu và xuất báo cáo tổng kết ra thư mục outputs/.

Cách sử dụng:
    python phase1_data_preparation.py
    python phase1_data_preparation.py --data-dir "path/to/custom/dataset"
    python phase1_data_preparation.py --generate-sample  # Tạo dữ liệu mẫu để chạy kiểm thử ngay lập tức
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cấu hình UTF-8 cho console Windows tránh UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

# Đảm bảo đường dẫn module src được nạp đúng
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import (
    DatasetInfo,
    discover_dataset_structure,
    load_and_validate_dataset,
)
from src.data_explorer import generate_exploration_report


# ==============================================================================
# HÀM TẠO DỮ LIỆU THỬ NGHIỆM MẪU (SYNTHETIC SAMPLE DATASET GENERATOR)
# ==============================================================================
def create_synthetic_sample_dataset(
    dest_dir: Path,
    splits: Optional[List[str]] = None,
    images_per_class: int = 8,
) -> Path:
    """
    Tạo bộ dữ liệu mẫu mô phỏng (Synthetic Dataset) với 3 loại da (Dry, Normal, Oily)
    để kiểm thử toàn diện toàn bộ pipeline xử lý ảnh số khi người dùng chưa kịp
    tải dataset thực tế từ Kaggle về máy.
    Generates realistic synthetic facial patches for test validation.

    Đặc trưng mô phỏng:
        - Dry Skin: Màu da hơi sẫm, độ tương phản vân bề mặt cao (khô nẻ), không có bóng sáng.
        - Normal Skin: Màu da hồng hào tự nhiên, phân bố độ sáng đồng đều, mịn màng.
        - Oily Skin: Có các vệt chói sáng phản xạ dầu nhờn (specular highlight: V > 230, S thấp).

    Args:
        dest_dir (Path): Thư mục đích tạo dữ liệu.
        splits (Optional[List[str]]): Danh sách các phân tập ['train', 'valid', 'test'].
        images_per_class (int): Số ảnh mẫu trên mỗi lớp da trong từng phân tập.

    Returns:
        Path: Đường dẫn thư mục gốc dữ liệu mẫu đã tạo.
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    print(f"\n[+] Đang khởi tạo bộ dữ liệu mẫu thử nghiệm tại: {dest_dir}...")
    np.random.seed(config.RANDOM_STATE)

    for split in splits:
        # Số ảnh tỉ lệ theo split: train nhiều nhất, valid và test ít hơn
        n_imgs = images_per_class if split == "train" else max(3, images_per_class // 2)

        for c_name in config.CLASS_NAMES:
            c_dir = dest_dir / split / c_name
            c_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n_imgs):
                # Kích thước ngẫu nhiên quanh 250 - 350 px mô phỏng ảnh thực tế
                h = np.random.randint(240, 320)
                w = np.random.randint(240, 320)

                # Màu nền sắc da cơ bản trong hệ BGR
                # B ~ 130-160, G ~ 160-190, R ~ 210-240
                base_b = np.random.randint(125, 155)
                base_g = np.random.randint(155, 185)
                base_r = np.random.randint(205, 235)

                img = np.zeros((h, w, 3), dtype=np.uint8)
                img[:, :, 0] = base_b
                img[:, :, 1] = base_g
                img[:, :, 2] = base_r

                # Thêm nhiễu Gaussian nhẹ tạo kết cấu da tự nhiên
                noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                if c_name == "dry":
                    # Da khô: Thêm các vết nứt nẻ / vân sần ráp (Fine texture, texture roughness)
                    texture = np.random.normal(0, 15, (h, w)).astype(np.int16)
                    for c_idx in range(3):
                        img[:, :, c_idx] = np.clip(
                            img[:, :, c_idx].astype(np.int16) + texture, 0, 255
                        ).astype(np.uint8)

                elif c_name == "oily":
                    # Da dầu: Thêm các vùng chói phản xạ bã nhờn (Specular Highlights)
                    # Tạo 2-4 vệt bóng sáng hình bầu dục
                    num_spots = np.random.randint(2, 5)
                    for _ in range(num_spots):
                        cx = np.random.randint(int(w * 0.25), int(w * 0.75))
                        cy = np.random.randint(int(h * 0.25), int(h * 0.75))
                        axes_len = (np.random.randint(15, 35), np.random.randint(10, 25))
                        angle = np.random.randint(0, 180)
                        # Vẽ elip phản chiếu ánh sáng trắng bão hòa
                        cv2.ellipse(
                            img,
                            (cx, cy),
                            axes_len,
                            angle,
                            0,
                            360,
                            (245, 248, 255),
                            -1,
                        )
                    # Làm mờ Gauss làm mượt viền bóng sáng
                    img = cv2.GaussianBlur(img, (9, 9), 3)

                elif c_name == "combination":
                    # Da hỗn hợp: Vệt bóng nhờn nhẹ vùng chữ T
                    num_spots = np.random.randint(1, 3)
                    for _ in range(num_spots):
                        cx = np.random.randint(int(w * 0.35), int(w * 0.65))
                        cy = np.random.randint(int(h * 0.2), int(h * 0.5))
                        axes_len = (np.random.randint(12, 22), np.random.randint(8, 16))
                        angle = np.random.randint(0, 180)
                        cv2.ellipse(
                            img,
                            (cx, cy),
                            axes_len,
                            angle,
                            0,
                            360,
                            (245, 248, 255),
                            -1,
                        )
                    img = cv2.GaussianBlur(img, (7, 7), 2)

                elif c_name == "normal":
                    # Da thường: Da mịn màng, làm mịn nhẹ
                    img = cv2.bilateralFilter(img, 5, 20, 20)

                # Lưu ảnh ra đĩa
                file_name = f"{c_name}_{split}_{i+1:03d}.jpg"
                file_path = c_dir / file_name
                # cv2.imwrite với unicode path
                cv2.imencode(".jpg", img)[1].tofile(str(file_path))

    print(f"[✓] Đã tạo thành công bộ dữ liệu mẫu mô phỏng tại {dest_dir}!\n")
    return dest_dir


# ==============================================================================
# HÀM HIỂN THỊ BẢNG TỔNG KẾT CONSOLE ĐẸP MẮT (PRETTY TERMINAL TABLES)
# ==============================================================================
def print_separator(char: str = "=", length: int = 78) -> None:
    print(char * length)


def print_section_header(title: str, step_num: int) -> None:
    print("\n" + "=" * 78)
    print(f"  BƯỚC {step_num}: {title.upper()}")
    print("=" * 78)


# ==============================================================================
# QUY TRÌNH CHÍNH CỦA GIAI ĐOẠN 1 (MAIN WORKFLOW)
# ==============================================================================
def run_phase1_pipeline(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    generate_sample_if_empty: bool = False,
    sample_quality_size: int = 500,
) -> DatasetInfo:
    """
    Thực thi toàn bộ chu trình Giai đoạn 1: Chuẩn bị dữ liệu xử lý ảnh số.
    Executes all Phase 1 data preparation steps end-to-end.

    Args:
        data_dir (Optional[Path]): Thư mục dataset.
        output_dir (Optional[Path]): Thư mục outputs.
        generate_sample_if_empty (bool): Tạo dữ liệu mẫu nếu chưa có ảnh nào.
        sample_quality_size (int): Số lượng ảnh lấy mẫu phân tích chất lượng sâu.

    Returns:
        DatasetInfo: Đối tượng chứa thông tin tổng hợp của Phase 1.
    """
    # Khởi tạo thư mục
    config.ensure_directories()
    out_dir = output_dir if output_dir else config.OUTPUTS_DIR
    target_data_dir = data_dir if data_dir else config.DATA_DIR

    print_separator("#")
    print("  ĐỒ ÁN XỬ LÝ ẢNH SỐ - PHASE 1: CHUẨN BỊ VÀ THẨM ĐỊNH BỘ DỮ LIỆU")
    print("  Đề tài: Phân loại loại da mặt (Dry, Normal, Oily Skin Types)")
    print_separator("#")

    # --------------------------------------------------------------------------
    # BƯỚC 1: KHÁM PHÁ VÀ NHẬN DIỆN CẤU TRÚC BỘ DỮ LIỆU
    # --------------------------------------------------------------------------
    print_section_header("Khám phá và nhận diện cấu trúc dataset", 1)
    structure = discover_dataset_structure(target_data_dir)

    # Nếu chưa có dữ liệu và người dùng cho phép tạo mẫu hoặc yêu cầu tạo mẫu
    if not structure.get("found", False) or structure.get("type") == "not_found":
        if generate_sample_if_empty:
            print("[!] Không tìm thấy dữ liệu hiện có. Đang tự động tạo dữ liệu mẫu mô phỏng...")
            create_synthetic_sample_dataset(target_data_dir, images_per_class=10)
            structure = discover_dataset_structure(target_data_dir)
        else:
            print(f"\n[!] CẢNH BÁO: Chưa tìm thấy dataset da mặt tại: {target_data_dir}")
            print("    Bộ dữ liệu chuẩn Kaggle 'Oily, Dry and Normal Skin Types' cần có cấu trúc:")
            print("    data/")
            print("    ├── train/{dry, normal, oily}")
            print("    ├── valid/{dry, normal, oily}")
            print("    └── test/{dry, normal, oily}")
            print("\n    Gợi ý giải pháp:")
            print(f"    1. Tải dataset và giải nén vào thư mục: {target_data_dir.resolve()}")
            print("    2. Hoặc chạy lại với cờ '--generate-sample' để tạo dữ liệu mô phỏng chạy thử nghiệm ngay:")
            print("       python phase1_data_preparation.py --generate-sample\n")
            # Tự động chuyển sang chế độ tạo mẫu để đảm bảo script kiểm thử thành công
            print("[+] Tự động bật chế độ --generate-sample để tiếp tục kiểm thử luồng...")
            create_synthetic_sample_dataset(target_data_dir, images_per_class=10)
            structure = discover_dataset_structure(target_data_dir)

    print(f"[*] Thư mục gốc bộ dữ liệu: {structure['root']}")
    print(f"[*] Cấu trúc phát hiện:     {structure['type'].upper()}")
    print(f"[*] Các lớp nhận diện:      {', '.join(structure['classes'])}")
    if structure.get("splits"):
        for s_name, s_path in structure["splits"].items():
            print(f"    - Tập {s_name.capitalize():<6}: {s_path}")

    # --------------------------------------------------------------------------
    # BƯỚC 2: TẢI VÀ THẨM ĐỊNH TÍNH TOÀN VẸN CÁC TỆP ẢNH
    # --------------------------------------------------------------------------
    print_section_header("Nạp và thẩm định tính toàn vẹn từng tệp ảnh", 2)
    dataset_info = load_and_validate_dataset(
        data_dir=structure["root"],
        auto_split_if_flat=True,
        compute_hashes=True,
        verbose=True,
    )
    df = dataset_info.df

    print(f"\n[*] Tổng số tệp ảnh quét được: {len(df)}")
    print(f"[*] Số ảnh hợp lệ (Valid):       {dataset_info.statistics.get('valid_images', 0)}")
    print(f"[*] Số ảnh lỗi hỏng (Corrupted): {dataset_info.statistics.get('invalid_images', 0)}")

    if dataset_info.statistics.get("invalid_images", 0) > 0:
        invalid_sample = dataset_info.get_invalid_df().head(5)
        print("\n[!] Mẫu các tệp ảnh bị lỗi phát hiện:")
        for _, row in invalid_sample.iterrows():
            print(f"    - {row['filename']}: {row['error_reason']}")

    # --------------------------------------------------------------------------
    # BƯỚC 3: THỐNG KÊ CHI TIẾT BỘ DỮ LIỆU
    # --------------------------------------------------------------------------
    print_section_header("Tổng hợp chỉ số thống kê bộ dữ liệu", 3)
    stats = dataset_info.statistics

    print("\n[+] Phân bố số lượng ảnh theo từng loại da (Class Balance):")
    print(f"    {'Loại da':<15} {'Tên hiển thị':<28} {'Số ảnh':<10} {'Tỉ lệ (%)':<10}")
    print("    " + "-" * 65)
    for c_name in config.CLASS_NAMES:
        c_count = stats.get("classes_counts", {}).get(c_name, 0)
        c_pct = stats.get("classes_percentages", {}).get(c_name, 0.0)
        c_vi = config.CLASS_LABELS_VI.get(c_name, c_name)
        print(f"    {c_name:<15} {c_vi:<28} {c_count:<10} {c_pct:<10.2f}%")

    if stats.get("dimensions"):
        dims = stats["dimensions"]
        print("\n[+] Thống kê độ phân giải không gian ảnh (Spatial Dimensions):")
        print(f"    - Chiều rộng (Width):   Min={dims['width_min']}px | Max={dims['width_max']}px | Mean={dims['width_mean']}px")
        print(f"    - Chiều cao (Height):  Min={dims['height_min']}px | Max={dims['height_max']}px | Mean={dims['height_mean']}px")
        print(f"    - Tỉ lệ W/H (Aspect):  Min={dims['aspect_ratio_min']} | Max={dims['aspect_ratio_max']} | Mean={dims['aspect_ratio_mean']}")

    if stats.get("file_size_kb"):
        fsize = stats["file_size_kb"]
        print("\n[+] Thống kê dung lượng lưu trữ (File Sizes):")
        print(f"    - Dung lượng tệp ảnh:   Min={fsize['min']}KB | Max={fsize['max']}KB | Mean={fsize['mean']}KB")
        print(f"    - Tổng dung lượng tập:  {fsize['total_mb']} MB")

    # --------------------------------------------------------------------------
    # BƯỚC 4: KIỂM TRA ẢNH TRÙNG LẶP (DUPLICATE DETECTION)
    # --------------------------------------------------------------------------
    print_section_header("Kiểm tra và bóc tách ảnh trùng lặp (Hashing)", 4)
    dup_sum = dataset_info.duplicates_summary
    exact_dups = dup_sum.get("exact_duplicates_count", 0)
    near_dups = dup_sum.get("near_duplicates_count", 0)

    print(f"[*] Ảnh trùng lặp tuyệt đối (Binary MD5):     {exact_dups} ảnh ({dup_sum.get('exact_groups_count', 0)} nhóm trùng)")
    print(f"[*] Ảnh tương đồng thị giác (Perceptual dHash): {near_dups} ảnh")

    if exact_dups > 0:
        print("[!] Phát hiện ảnh trùng lặp! Danh sách nhóm trùng được ghi nhận trong metadata.")

    # --------------------------------------------------------------------------
    # BƯỚC 5: PHÂN TÍCH CHẤT LƯỢNG ẢNH & PHÁT HIỆN ẢNH VẤN ĐỀ
    # --------------------------------------------------------------------------
    print_section_header("Phân tích chất lượng ảnh số & Nhận diện lỗi", 5)
    print("[*] Đang đo lường: Độ sắc nét Laplacian, Độ sáng trung bình, Vùng chói nhờn...")
    report_res = generate_exploration_report(
        dataset_info=dataset_info,
        output_dir=out_dir,
        sample_quality_size=sample_quality_size,
    )
    prob_summary = report_res["problematic_summary"]

    print(f"[*] Ảnh bị mờ nét (Laplacian Var < {config.QUALITY_THRESHOLDS['blur_laplacian_var']}): {prob_summary.get('blurry_count', 0)}")
    print(f"[*] Ảnh quá tối (Brightness < {config.QUALITY_THRESHOLDS['dark_intensity']}):        {prob_summary.get('too_dark_count', 0)}")
    print(f"[*] Ảnh quá sáng (Brightness > {config.QUALITY_THRESHOLDS['bright_intensity']}):     {prob_summary.get('too_bright_count', 0)}")
    print(f"[*] Tổng số ảnh cảnh báo chất lượng:                                {prob_summary.get('total_problematic', 0)}")

    # --------------------------------------------------------------------------
    # BƯỚC 6: KIỂM ĐỊNH PHÂN CHIA TẬP DỮ LIỆU (TRAIN / VALID / TEST)
    # --------------------------------------------------------------------------
    print_section_header("Kiểm định phân chia phân tập dữ liệu (Splits)", 6)
    split_counts = dataset_info.df["split"].value_counts().to_dict()
    print("[+] Phân bố số lượng ảnh trên các tập:")
    for s_name, s_count in split_counts.items():
        pct = (s_count / len(dataset_info.df) * 100) if len(dataset_info.df) > 0 else 0
        print(f"    - Tập {s_name.capitalize():<8}: {s_count:<6} ({pct:.1f}%)")

    # Bảng phân tầng chéo giữa Split và Class
    cross = pd.crosstab(dataset_info.df["split"], dataset_info.df["class_name"])
    print("\n[+] Bảng phân tầng chi tiết (Stratified Cross-Tabulation):")
    print(cross.to_string())

    # --------------------------------------------------------------------------
    # BƯỚC 7: XUẤT TOÀN BỘ KẾT QUẢ VÀ BÁO CÁO TỔNG KẾT
    # --------------------------------------------------------------------------
    print_section_header("Xuất kết quả đầu ra & Báo cáo tổng kết", 7)

    # Lưu bảng metadata hoàn chỉnh
    meta_path = out_dir / "dataset_metadata.csv"
    dataset_info.save_metadata(meta_path)
    print(f"[✓] Đã lưu bảng Metadata:  {meta_path}")

    # Lưu file JSON tóm tắt
    json_path = out_dir / "phase1_dataset_summary.json"
    dataset_info.save_summary_json(json_path)
    print(f"[✓] Đã lưu tệp JSON thống kê: {json_path}")

    # Báo cáo Markdown
    print(f"[✓] Đã tạo Báo cáo Markdown:  {report_res['report_md_path']}")

    # Các biểu đồ đã kết xuất
    print("[✓] Các biểu đồ trực quan hóa đã xuất bản:")
    for fig_path in report_res["figures"]:
        print(f"    - {fig_path}")

    print_separator("=")
    print("  GIAI ĐOẠN 1 (PHASE 1: DATA PREPARATION) ĐÃ HOÀN TẤT THÀNH CÔNG!")
    print("  Toàn bộ dữ liệu đã sẵn sàng cho Giai đoạn 2: Tiền xử lý & Trích xuất Đặc trưng.")
    print_separator("=")

    return dataset_info


# ==============================================================================
# ĐIỂM VÀO KỊCH BẢN (ENTRY POINT)
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Data Preparation & Exploration for Skin Types Classification"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Đường dẫn tới thư mục bộ dữ liệu (Mặc định: config.DATA_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Đường dẫn thư mục lưu kết quả (Mặc định: config.OUTPUTS_DIR)",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Tạo dữ liệu thử nghiệm mô phỏng nếu chưa có dataset thực tế",
    )
    parser.add_argument(
        "--sample-quality-size",
        type=int,
        default=500,
        help="Số lượng ảnh tối đa để phân tích chất lượng sâu (mặc định: 500)",
    )

    args = parser.parse_args()

    data_dir_path = Path(args.data_dir).resolve() if args.data_dir else None
    output_dir_path = Path(args.output_dir).resolve() if args.output_dir else None

    run_phase1_pipeline(
        data_dir=data_dir_path,
        output_dir=output_dir_path,
        generate_sample_if_empty=args.generate_sample,
        sample_quality_size=args.sample_quality_size,
    )


if __name__ == "__main__":
    main()
