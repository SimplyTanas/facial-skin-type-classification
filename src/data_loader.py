"""
Mô-đun Nạp và Tiền kiểm tra Dữ liệu (Data Loader & Validation Module)
==================================================================
Đề tài: Phân loại loại da mặt (Dry, Normal, Oily) sử dụng Kỹ thuật Xử lý ảnh số & Học máy.

Chức năng chính:
    1. discover_dataset_structure: Tự động nhận diện cấu trúc dataset Kaggle (train/valid/test hoặc flat).
    2. load_dataset_paths: Duyệt cây thư mục và lập chỉ mục toàn bộ đường dẫn ảnh và nhãn.
    3. validate_image: Kiểm tra tính toàn vẹn của từng ảnh (không bị lỗi, kích thước > 0, đọc được bằng OpenCV/PIL).
    4. detect_duplicate_images: Phát hiện ảnh trùng lặp tuyệt đối (MD5) và trùng lặp xấp xỉ (dHash).
    5. split_dataset_stratified: Phân tầng dữ liệu thành train/valid/test với tỉ lệ chuẩn xác.
    6. get_dataset_statistics: Tổng hợp các chỉ số thống kê kích thước, định dạng, độ phân giải.
    7. DatasetInfo: Lớp chứa metadata toàn diện của bộ dữ liệu.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Nhập cấu hình từ src.config
try:
    from . import config
except ImportError:
    import config


# ==============================================================================
# HÀM BĂM ẢNH ĐỂ PHÁT HIỆN TRÙNG LẶP (HASHING FUNCTIONS)
# ==============================================================================
def compute_md5(file_path: Union[Path, str], chunk_size: int = 65536) -> str:
    """
    Tính mã băm MD5 từ nội dung nhị phân của tệp ảnh để tìm ảnh trùng lặp 100%.
    Computes MD5 checksum of an image file.

    Args:
        file_path (Path | str): Đường dẫn tới tệp ảnh.
        chunk_size (int): Kích thước khối đọc dữ liệu nhị phân (bytes).

    Returns:
        str: Chuỗi hex biểu diễn mã băm MD5.
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def compute_dhash(image_path: Union[Path, str], hash_size: int = 8) -> Optional[str]:
    """
    Tính mã băm cảm nhận (Difference Hash - dHash) để tìm ảnh thị giác tương đồng
    hoặc ảnh bị nén lại/đổi kích thước nhưng nội dung gần như giữ nguyên.
    Computes Difference Hash (dHash) for perceptual duplicate detection.

    Thuật toán:
        1. Chuyển ảnh về ảnh xám (Grayscale).
        2. Resize về kích thước (hash_size + 1, hash_size), mặc định (9, 8).
        3. So sánh độ sáng pixel liền kề theo hàng ngang (pixel[col] > pixel[col+1]).
        4. Tạo chuỗi bit 64-bit và chuyển đổi sang dạng chuỗi Hexadecimal.

    Args:
        image_path (Path | str): Đường dẫn tới tệp ảnh.
        hash_size (int): Kích thước lưới so sánh (mặc định 8 cho hash 64-bit).

    Returns:
        Optional[str]: Chuỗi hex của dHash, hoặc None nếu không đọc được ảnh.
    """
    try:
        with Image.open(image_path) as img:
            # Chuyển đổi về ảnh xám và resize
            gray = img.convert("L").resize(
                (hash_size + 1, hash_size), Image.Resampling.BILINEAR
            )
            pixels = np.array(gray, dtype=np.int32)
            # So sánh pixel liền kề: hàng ngang
            diff = pixels[:, 1:] > pixels[:, :-1]
            # Biến ma trận boolean thành số nguyên nhị phân
            decimal_value = 0
            for bit in diff.flatten():
                decimal_value = (decimal_value << 1) | int(bit)
            return f"{decimal_value:016x}"
    except Exception:
        return None


# ==============================================================================
# KIỂM TRA TÍNH HỢP LỆ CỦA ẢNH (IMAGE VALIDATION)
# ==============================================================================
def validate_single_image(
    image_path: Union[Path, str]
) -> Tuple[bool, Optional[str], Tuple[int, int, int], float]:
    """
    Kiểm tra một tệp ảnh có đọc được an toàn, không bị lỗi hỏng (corrupted) hay không.
    Validates that a single image can be safely opened and read.

    Args:
        image_path (Path | str): Đường dẫn tới tệp ảnh cần kiểm tra.

    Returns:
        Tuple chứa:
            - is_valid (bool): True nếu ảnh đọc bình thường.
            - error_reason (Optional[str]): Lý do lỗi nếu không hợp lệ.
            - (width, height, channels) (Tuple[int, int, int]): Kích thước ảnh.
            - file_size_kb (float): Dung lượng tệp ảnh tính theo Kilobytes.
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        return False, "File does not exist", (0, 0, 0), 0.0

    file_size_kb = path_obj.stat().st_size / 1024.0
    if file_size_kb <= 0.0:
        return False, "Empty file (0 bytes)", (0, 0, 0), 0.0

    # 1. Kiểm tra bằng Pillow
    try:
        with Image.open(path_obj) as img:
            img.verify()
    except Exception as exc:
        return False, f"PIL Verification Error: {str(exc)}", (0, 0, 0), file_size_kb

    # 2. Đọc thử dữ liệu mảng numpy bằng OpenCV
    try:
        # cv2.imread với đường dẫn Unicode an toàn trên Windows
        img_cv = cv2.imdecode(
            np.fromfile(str(path_obj), dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if img_cv is None:
            return False, "OpenCV failed to decode image", (0, 0, 0), file_size_kb

        if img_cv.ndim == 2:
            height, width = img_cv.shape
            channels = 1
        elif img_cv.ndim == 3:
            height, width, channels = img_cv.shape
        else:
            return (
                False,
                f"Unsupported array dimensions: {img_cv.ndim}",
                (0, 0, 0),
                file_size_kb,
            )

        if width < config.QUALITY_THRESHOLDS["min_dimension"] or height < config.QUALITY_THRESHOLDS["min_dimension"]:
            return (
                False,
                f"Image dimension ({width}x{height}) is below minimum threshold {config.QUALITY_THRESHOLDS['min_dimension']}",
                (width, height, channels),
                file_size_kb,
            )

        return True, None, (width, height, channels), file_size_kb

    except Exception as exc:
        return False, f"OpenCV Decode Exception: {str(exc)}", (0, 0, 0), file_size_kb


# ==============================================================================
# TỰ ĐỘNG KHÁM PHÁ CẤU TRÚC BỘ DỮ LIỆU (DATASET DISCOVERY)
# ==============================================================================
def discover_dataset_structure(
    base_path: Optional[Union[Path, str]] = None
) -> Dict[str, Any]:
    """
    Tự động quét và phát hiện vị trí cũng như cấu trúc của bộ dữ liệu phân loại da.
    Automatically detects dataset location and organizational structure.

    Hỗ trợ cả hai dạng cấu trúc:
        1. Cấu trúc chuẩn Kaggle (Đã chia tập):
           base_path/
           ├── train/{dry, normal, oily}
           ├── valid/ hoặc validation/ hoặc val/{dry, normal, oily}
           └── test/{dry, normal, oily}
        2. Cấu trúc phẳng (Chưa chia tập):
           base_path/{dry, normal, oily}

    Args:
        base_path (Optional[Path | str]): Thư mục tìm kiếm. Nếu None, sẽ tự động tìm
            trong config.DATA_DIR và các thư mục ứng viên khả dĩ.

    Returns:
        Dict[str, Any]: Thông tin cấu trúc phát hiện được gồm:
            - 'root': Path thư mục gốc của dataset.
            - 'type': 'split' (đã chia) hoặc 'flat' (phẳng) hoặc 'not_found'.
            - 'splits': Dict ánh xạ 'train', 'valid', 'test' -> Path tương ứng.
            - 'classes': Danh sách các tên lớp phát hiện được.
            - 'found': bool.
    """
    candidate_roots: List[Path] = []
    if base_path is not None:
        candidate_roots.append(Path(base_path).resolve())
    else:
        candidate_roots.extend([
            config.DATA_DIR,
            config.PROJECT_ROOT / "data",
            config.PROJECT_ROOT / "dataset",
            config.PROJECT_ROOT / "Oily-Dry-Skin-Types",
            config.PROJECT_ROOT / "skin_types",
            config.PROJECT_ROOT,
        ])

    valid_class_keywords = {"dry", "normal", "oily", "combination"}

    for root in candidate_roots:
        if not root.exists() or not root.is_dir():
            continue

        # Kiểm tra xem có cấu trúc train / valid / test hay không
        subdirs = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
        has_train = "train" in subdirs
        has_valid = any(k in subdirs for k in ("valid", "validation", "val"))
        has_test = "test" in subdirs

        valid_key = next((k for k in ("valid", "validation", "val") if k in subdirs), None)

        if has_train:
            splits_dict: Dict[str, Path] = {"train": subdirs["train"]}
            if valid_key:
                splits_dict["valid"] = subdirs[valid_key]
            if has_test:
                splits_dict["test"] = subdirs["test"]

            # Tìm danh sách classes bên trong thư mục train
            train_classes = [
                p.name.lower()
                for p in splits_dict["train"].iterdir()
                if p.is_dir() and p.name.lower() in valid_class_keywords
            ]
            if not train_classes:
                # Nếu không khớp chính xác keyword, lấy toàn bộ thư mục con
                train_classes = [
                    p.name.lower() for p in splits_dict["train"].iterdir() if p.is_dir()
                ]

            if train_classes:
                return {
                    "root": root,
                    "type": "split",
                    "splits": splits_dict,
                    "classes": sorted(list(train_classes)),
                    "found": True,
                }

        # Kiểm tra nếu là cấu trúc phẳng: root/ chứa trực tiếp {dry, normal, oily}
        flat_classes = [
            name for name in subdirs.keys() if name in valid_class_keywords
        ]
        if len(flat_classes) >= 2:
            return {
                "root": root,
                "type": "flat",
                "splits": {},
                "classes": sorted(flat_classes),
                "found": True,
            }

    # Nếu không tìm thấy cấu trúc hợp lệ nào
    default_root = Path(base_path) if base_path else config.DATA_DIR
    return {
        "root": default_root,
        "type": "not_found",
        "splits": {},
        "classes": config.CLASS_NAMES,
        "found": False,
    }


# ==============================================================================
# LẬP DANH MỤC ĐƯỜNG DẪN ẢNH VÀ GẮN NHÃN (LOAD PATHS & LABELS)
# ==============================================================================
def load_dataset_records(
    structure: Dict[str, Any],
    valid_extensions: Tuple[str, ...] = config.VALID_IMAGE_EXTENSIONS,
) -> pd.DataFrame:
    """
    Duyệt toàn bộ tệp ảnh trong các thư mục theo cấu trúc dataset đã phát hiện
    và tạo DataFrame chứa thông tin định danh ban đầu.
    Crawls dataset structure and constructs an initial catalog DataFrame.

    Args:
        structure (Dict[str, Any]): Cấu trúc dataset trả về từ discover_dataset_structure.
        valid_extensions (Tuple[str, ...]): Các đuôi tệp mở rộng hợp lệ.

    Returns:
        pd.DataFrame: Bảng chứa các cột:
            ['image_path', 'filename', 'class_name', 'label', 'split', 'file_ext']
    """
    records: List[Dict[str, Any]] = []

    if not structure.get("found", False):
        return pd.DataFrame(
            columns=["image_path", "filename", "class_name", "label", "split", "file_ext"]
        )

    struct_type = structure.get("type", "not_found")
    classes = structure.get("classes", config.CLASS_NAMES)

    if struct_type == "split":
        splits_dict: Dict[str, Path] = structure["splits"]
        for split_name, split_dir in splits_dict.items():
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                c_name = class_dir.name.lower()
                if c_name not in classes:
                    continue
                label_idx = config.CLASS_TO_IDX.get(c_name, -1)

                for file_path in class_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                        records.append({
                            "image_path": str(file_path.resolve()),
                            "filename": file_path.name,
                            "class_name": c_name,
                            "label": label_idx,
                            "split": split_name,
                            "file_ext": file_path.suffix.lower(),
                        })

    elif struct_type == "flat":
        root_dir: Path = structure["root"]
        for class_dir in root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            c_name = class_dir.name.lower()
            if c_name not in classes:
                continue
            label_idx = config.CLASS_TO_IDX.get(c_name, -1)

            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    records.append({
                        "image_path": str(file_path.resolve()),
                        "filename": file_path.name,
                        "class_name": c_name,
                        "label": label_idx,
                        "split": "unassigned",
                        "file_ext": file_path.suffix.lower(),
                    })

    df = pd.DataFrame(records)
    return df


def load_dataset_paths(
    base_path: Optional[Union[Path, str]] = None,
    valid_extensions: Tuple[str, ...] = config.VALID_IMAGE_EXTENSIONS,
) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Quét và trả về từ điển chứa danh sách tuple (image_path, label) theo từng tập split
    ('train', 'valid', 'test', 'all').
    Crawls and returns image paths categorized by split and class label.

    Args:
        base_path (Optional[Path | str]): Thư mục tìm kiếm dữ liệu.
        valid_extensions (Tuple[str, ...]): Các đuôi tệp ảnh hợp lệ.

    Returns:
        Dict[str, List[Tuple[Path, str]]]: Từ điển mapping split -> danh sách (Path, label_str).
    """
    structure = discover_dataset_structure(base_path)
    df = load_dataset_records(structure, valid_extensions=valid_extensions)
    result: Dict[str, List[Tuple[Path, str]]] = {
        "train": [],
        "valid": [],
        "test": [],
        "all": [],
    }
    if df.empty:
        return result

    for _, row in df.iterrows():
        p = Path(row["image_path"])
        lbl = str(row["class_name"])
        split = str(row["split"])
        item = (p, lbl)
        result["all"].append(item)
        if split in result:
            result[split].append(item)
        else:
            result[split] = [item]

    return result


# ==============================================================================
# PHÂN CHIA DỮ LIỆU CÓ PHÂN TẦNG (STRATIFIED DATASET SPLIT)
# ==============================================================================
def split_dataset_stratified(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Phân chia tập dữ liệu thành Train / Valid / Test theo phương pháp phân tầng (Stratified)
    đảm bảo tỉ lệ nhãn các lớp da đồng đều giữa các tập.
    Stratifies and splits dataset into train, valid, and test partitions.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu ảnh và cột 'class_name' hoặc 'label'.
        train_ratio (float): Tỉ lệ tập Train (mặc định 0.70 = 70%).
        valid_ratio (float): Tỉ lệ tập Validation (mặc định 0.15 = 15%).
        test_ratio (float): Tỉ lệ tập Test (mặc định 0.15 = 15%).
        random_state (int): Hạt giống ngẫu nhiên đảm bảo tính tái lập (reproducibility).

    Returns:
        pd.DataFrame: DataFrame có cột 'split' được cập nhật ('train', 'valid', 'test').
    """
    total_ratio = train_ratio + valid_ratio + test_ratio
    assert np.isclose(total_ratio, 1.0), "Tổng các tỉ lệ phân chia phải bằng 1.0"

    df_result = df.copy()
    if len(df_result) == 0:
        return df_result

    # Chia Train vs (Valid + Test)
    temp_ratio = valid_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        df_result.index,
        test_size=temp_ratio,
        stratify=df_result["class_name"],
        random_state=random_state,
    )

    # Chia tiếp phần temp thành Valid và Test
    valid_fraction_of_temp = valid_ratio / temp_ratio
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - valid_fraction_of_temp),
        stratify=df_result.loc[temp_idx, "class_name"],
        random_state=random_state,
    )

    df_result.loc[train_idx, "split"] = "train"
    df_result.loc[valid_idx, "split"] = "valid"
    df_result.loc[test_idx, "split"] = "test"

    return df_result


# ==============================================================================
# PHÁT HIỆN ẢNH TRÙNG LẶP (DUPLICATE DETECTION)
# ==============================================================================
def detect_duplicate_images(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Phát hiện các ảnh bị trùng lặp dựa trên băm mã nhị phân tuyệt đối (MD5)
    và băm cảm nhận thị giác (dHash).
    Identifies exact and perceptual duplicate images in the dataset.

    Args:
        df (pd.DataFrame): DataFrame có chứa cột 'md5_hash' và 'dhash'.

    Returns:
        Tuple:
            - duplicates_df (pd.DataFrame): Danh sách các hàng ảnh bị trùng lặp.
            - summary (Dict[str, Any]): Thống kê tổng số ảnh trùng và các nhóm trùng.
    """
    if "md5_hash" not in df.columns or len(df) == 0:
        return pd.DataFrame(), {"exact_duplicates_count": 0, "near_duplicates_count": 0}

    # 1. Trùng lặp tuyệt đối (Exact MD5 Duplicates)
    exact_mask = df.duplicated(subset=["md5_hash"], keep=False)
    exact_duplicates = df[exact_mask].sort_values(by="md5_hash")

    # 2. Trùng lặp cảm nhận thị giác (Near dHash Duplicates)
    # Loại trừ những ảnh đã bị trùng exact để tránh đếm lặp
    valid_dhash_df = df[df["dhash"].notna()]
    near_mask = valid_dhash_df.duplicated(subset=["dhash"], keep=False) & ~exact_mask
    near_duplicates = valid_dhash_df[near_mask].sort_values(by="dhash")

    all_duplicates = df[exact_mask | (df.index.isin(near_duplicates.index))].copy()

    # Nhóm các nhóm trùng lặp
    md5_groups = exact_duplicates.groupby("md5_hash")["image_path"].apply(list).to_dict()
    # Chỉ giữ các nhóm có > 1 phần tử
    md5_groups = {k: v for k, v in md5_groups.items() if len(v) > 1}

    summary = {
        "exact_duplicates_count": int(exact_mask.sum()),
        "exact_groups_count": len(md5_groups),
        "near_duplicates_count": int(near_mask.sum()),
        "total_problematic_duplicates": int(len(all_duplicates)),
        "md5_duplicate_groups": md5_groups,
    }

    return all_duplicates, summary


# ==============================================================================
# TỔNG HỢP THỐNG KÊ DATASET (DATASET STATISTICS)
# ==============================================================================
def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Tính toán toàn diện các chỉ số thống kê về bộ dữ liệu da mặt.
    Computes comprehensive dataset statistics (class counts, dimensions, file sizes).

    Args:
        df (pd.DataFrame): DataFrame chứa metadata các ảnh.

    Returns:
        Dict[str, Any]: Từ điển thống kê chi tiết.
    """
    if len(df) == 0:
        return {
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "classes_distribution": {},
            "splits_distribution": {},
            "cross_distribution": {},
        }

    total_images = len(df)
    valid_count = int(df["is_valid"].sum()) if "is_valid" in df.columns else total_images
    invalid_count = total_images - valid_count

    # Thống kê theo Lớp (Class)
    class_counts = df["class_name"].value_counts().to_dict()
    class_pcts = (df["class_name"].value_counts(normalize=True) * 100).round(2).to_dict()

    # Thống kê theo Phân tập (Split)
    split_counts = df["split"].value_counts().to_dict()

    # Thống kê chéo (Split x Class)
    cross_tab = (
        pd.crosstab(df["split"], df["class_name"]).to_dict(orient="index")
        if "split" in df.columns
        else {}
    )

    # Thống kê định dạng tệp (Extensions)
    ext_counts = df["file_ext"].value_counts().to_dict() if "file_ext" in df.columns else {}

    # Thống kê kích thước và dung lượng tệp
    stats: Dict[str, Any] = {
        "total_images": total_images,
        "valid_images": valid_count,
        "invalid_images": invalid_count,
        "classes_counts": class_counts,
        "classes_percentages": class_pcts,
        "splits_counts": split_counts,
        "cross_split_class": cross_tab,
        "extensions_counts": ext_counts,
    }

    if "width" in df.columns and "height" in df.columns and valid_count > 0:
        valid_df = df[df["is_valid"]] if "is_valid" in df.columns else df
        stats["dimensions"] = {
            "width_min": int(valid_df["width"].min()),
            "width_max": int(valid_df["width"].max()),
            "width_mean": float(round(valid_df["width"].mean(), 2)),
            "width_median": float(valid_df["width"].median()),
            "height_min": int(valid_df["height"].min()),
            "height_max": int(valid_df["height"].max()),
            "height_mean": float(round(valid_df["height"].mean(), 2)),
            "height_median": float(valid_df["height"].median()),
            "aspect_ratio_mean": float(round(valid_df["aspect_ratio"].mean(), 2)),
            "aspect_ratio_min": float(round(valid_df["aspect_ratio"].min(), 2)),
            "aspect_ratio_max": float(round(valid_df["aspect_ratio"].max(), 2)),
        }

    if "file_size_kb" in df.columns and valid_count > 0:
        valid_df = df[df["is_valid"]] if "is_valid" in df.columns else df
        stats["file_size_kb"] = {
            "min": float(round(valid_df["file_size_kb"].min(), 2)),
            "max": float(round(valid_df["file_size_kb"].max(), 2)),
            "mean": float(round(valid_df["file_size_kb"].mean(), 2)),
            "median": float(round(valid_df["file_size_kb"].median(), 2)),
            "total_mb": float(round(valid_df["file_size_kb"].sum() / 1024.0, 2)),
        }

    return stats


# ==============================================================================
# LỚP ĐẠI DIỆN METADATA DATASET (DATASET INFO CLASS)
# ==============================================================================
@dataclass
class DatasetInfo:
    """
    Lớp dữ liệu (Dataclass) lưu trữ tập trung toàn bộ siêu dữ liệu, danh sách ảnh,
    kết quả kiểm tra tính hợp lệ và thống kê của bộ dữ liệu da mặt.
    Encapsulates all metadata and data frames for the facial skin types dataset.
    """

    root_path: Path
    structure_type: str
    classes: List[str] = field(default_factory=lambda: list(config.CLASS_NAMES))
    splits: Dict[str, Path] = field(default_factory=dict)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicates_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicates_summary: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def get_valid_df(self) -> pd.DataFrame:
        """Trả về DataFrame chỉ gồm các ảnh hợp lệ."""
        if "is_valid" in self.df.columns:
            return self.df[self.df["is_valid"]].copy()
        return self.df.copy()

    def get_invalid_df(self) -> pd.DataFrame:
        """Trả về DataFrame gồm các ảnh bị lỗi/hỏng."""
        if "is_valid" in self.df.columns:
            return self.df[~self.df["is_valid"]].copy()
        return pd.DataFrame()

    def save_metadata(self, csv_path: Union[Path, str]) -> Path:
        """
        Lưu toàn bộ bảng metadata ra tệp CSV để tái sử dụng trong các Phase tiếp theo.
        Saves metadata DataFrame to a CSV file.
        """
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(p, index=False, encoding="utf-8")
        return p

    def save_summary_json(self, json_path: Union[Path, str]) -> Path:
        """
        Lưu báo cáo thống kê ra định dạng JSON.
        Saves summary statistics dictionary to JSON.
        """
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "root_path": str(self.root_path),
            "structure_type": self.structure_type,
            "classes": self.classes,
            "splits": {k: str(v) for k, v in self.splits.items()},
            "statistics": self.statistics,
            "duplicates_summary": self.duplicates_summary,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=4, ensure_ascii=False)
        return p


# ==============================================================================
# HÀM TẢI VÀ TIỀN KIỂM TOÀN BỘ DATASET (MAIN LOADING FUNCTION)
# ==============================================================================
def load_and_validate_dataset(
    data_dir: Optional[Union[Path, str]] = None,
    auto_split_if_flat: bool = True,
    compute_hashes: bool = True,
    verbose: bool = True,
) -> DatasetInfo:
    """
    Quy trình tích hợp hoàn chỉnh nạp và tiền kiểm tra dữ liệu:
        1. Khám phá cấu trúc dataset (discover_dataset_structure).
        2. Tải danh sách toàn bộ ảnh (load_dataset_records).
        3. Kiểm tra tính hợp lệ từng ảnh (validate_single_image).
        4. Tính mã băm phát hiện trùng lặp nếu compute_hashes=True.
        5. Tự động chia tập train/valid/test nếu cấu trúc ở dạng phẳng (flat).
        6. Tính toán thống kê toàn diện (get_dataset_statistics).

    Args:
        data_dir (Optional[Path | str]): Thư mục dataset. Mặc định là config.DATA_DIR.
        auto_split_if_flat (bool): Tự động phân chia train/val/test nếu dataset phẳng.
        compute_hashes (bool): Có tính MD5 và dHash để tìm duplicate hay không.
        verbose (bool): In thông báo tiến trình ra màn hình console.

    Returns:
        DatasetInfo: Đối tượng chứa dữ liệu đã được nạp và phân tích hoàn chỉnh.
    """
    if verbose:
        print("[+] Bước 1: Quét và nhận diện cấu trúc thư mục dữ liệu...")

    structure = discover_dataset_structure(data_dir)
    root_path = Path(structure["root"])

    if verbose:
        print(f"    - Thư mục gốc: {root_path}")
        print(f"    - Loại cấu trúc: {structure['type']}")
        print(f"    - Các lớp nhận diện: {structure['classes']}")

    # Nạp danh sách bản ghi
    df = load_dataset_records(structure)
    total_found = len(df)

    if total_found == 0:
        if verbose:
            print(f"    [!] CẢNH BÁO: Không tìm thấy ảnh hợp lệ nào trong {root_path}!")
        empty_info = DatasetInfo(
            root_path=root_path,
            structure_type=structure["type"],
            classes=structure["classes"],
            splits=structure.get("splits", {}),
            df=df,
            statistics=get_dataset_statistics(df),
        )
        return empty_info

    if verbose:
        print(f"[+] Bước 2: Kiểm tra tính hợp lệ cho {total_found} ảnh...")

    # Duyệt và tiền kiểm tra
    is_valid_list = []
    error_reasons = []
    widths = []
    heights = []
    channels_list = []
    aspect_ratios = []
    sizes_kb = []
    md5_hashes = []
    dhashes = []

    for idx, row in df.iterrows():
        img_path = Path(row["image_path"])
        valid, reason, dims, size_kb = validate_single_image(img_path)
        is_valid_list.append(valid)
        error_reasons.append(reason)
        w, h, c = dims
        widths.append(w)
        heights.append(h)
        channels_list.append(c)
        aspect_ratios.append(round(w / h, 3) if h > 0 else 0.0)
        sizes_kb.append(round(size_kb, 2))

        if compute_hashes and valid:
            md5_hashes.append(compute_md5(img_path))
            dhashes.append(compute_dhash(img_path))
        else:
            md5_hashes.append(None)
            dhashes.append(None)

    df["is_valid"] = is_valid_list
    df["error_reason"] = error_reasons
    df["width"] = widths
    df["height"] = heights
    df["channels"] = channels_list
    df["aspect_ratio"] = aspect_ratios
    df["file_size_kb"] = sizes_kb

    if compute_hashes:
        df["md5_hash"] = md5_hashes
        df["dhash"] = dhashes

    # Nếu cấu trúc phẳng và người dùng muốn chia train/valid/test
    if structure["type"] == "flat" and auto_split_if_flat:
        if verbose:
            print("[+] Tự động phân chia Stratified Train/Valid/Test (70/15/15)...")
        # Chỉ chia trên các ảnh hợp lệ
        valid_mask = df["is_valid"]
        df_valid_split = split_dataset_stratified(
            df[valid_mask],
            train_ratio=config.DEFAULT_SPLIT_RATIOS[0],
            valid_ratio=config.DEFAULT_SPLIT_RATIOS[1],
            test_ratio=config.DEFAULT_SPLIT_RATIOS[2],
            random_state=config.RANDOM_STATE,
        )
        df.loc[valid_mask, "split"] = df_valid_split["split"]

    # Phát hiện ảnh trùng lặp
    duplicates_df = pd.DataFrame()
    duplicates_summary = {}
    if compute_hashes:
        if verbose:
            print("[+] Bước 3: Phát hiện ảnh trùng lặp qua MD5 và dHash...")
        duplicates_df, duplicates_summary = detect_duplicate_images(df[df["is_valid"]])

    # Tính toán thống kê
    stats = get_dataset_statistics(df)

    if verbose:
        print(f"[+] Hoàn tất nạp dữ liệu! Tổng ảnh: {len(df)} | Hợp lệ: {stats.get('valid_images', 0)} | Bị lỗi: {stats.get('invalid_images', 0)}")

    dataset_info = DatasetInfo(
        root_path=root_path,
        structure_type=structure["type"],
        classes=structure["classes"],
        splits=structure.get("splits", {}),
        df=df,
        duplicates_df=duplicates_df,
        duplicates_summary=duplicates_summary,
        statistics=stats,
    )

    return dataset_info
