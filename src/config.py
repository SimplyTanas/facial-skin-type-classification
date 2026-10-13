"""
Cấu hình tập trung cho Toàn bộ Dự án Xử lý Ảnh số (Central Project Configuration)
=============================================================================
Đề tài: Phân loại loại da mặt (Dry, Normal, Oily) sử dụng Kỹ thuật Xử lý ảnh số & Học máy.
Mục tiêu:
    - Quản lý tập trung toàn bộ siêu tham số (hyperparameters), đường dẫn thư mục.
    - Cung cấp các ngưỡng lọc da (skin segmentation) cho các không gian màu HSV, YCbCr.
    - Định nghĩa các tham số trích xuất đặc trưng kết cấu (LBP, GLCM) và màu sắc (Color Moments).
    - Cấu hình dải tham số tìm kiếm tối ưu cho các mô hình phân loại (KNN, SVM).
"""

from pathlib import Path
import os
from typing import Dict, List, Tuple
import numpy as np

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC (PROJECT PATHS)
# ==============================================================================
# Thư mục gốc của project (Root Directory: .../XLA)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Thư mục chứa dữ liệu (Dataset Directory)
# Người dùng có thể ghi đè qua biến môi trường 'XLA_DATA_DIR' nếu muốn
DATA_DIR: Path = Path(os.environ.get("XLA_DATA_DIR", PROJECT_ROOT / "data"))

# Các phân tập dữ liệu chuẩn theo cấu trúc Kaggle Dataset:
# dataset_root/
# ├── train/ (dry, normal, oily)
# ├── valid/ hoặc validation/ (dry, normal, oily)
# └── test/  (dry, normal, oily)
TRAIN_DIR: Path = DATA_DIR / "train"
VALID_DIR: Path = DATA_DIR / "valid"
TEST_DIR: Path = DATA_DIR / "test"

# Thư mục lưu trữ kết quả đầu ra
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
PROCESSED_DATA_DIR: Path = OUTPUTS_DIR / "processed_data"

# ==============================================================================
# 2. CẤU HÌNH NHÃN VÀ LỚP PHÂN LOẠI (CLASS LABELS & MAPPINGS)
# ==============================================================================
CLASS_NAMES: List[str] = ["combination", "dry", "normal", "oily"]
NUM_CLASSES: int = len(CLASS_NAMES)

# Ánh xạ từ tên lớp sang chỉ số số nguyên (Class Name -> Index)
CLASS_TO_IDX: Dict[str, int] = {
    "combination": 0,
    "dry": 1,
    "normal": 2,
    "oily": 3,
}

# Ánh xạ từ chỉ số số nguyên về tên lớp (Index -> Class Name)
IDX_TO_CLASS: Dict[int, str] = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Nhãn tiếng Việt phục vụ hiển thị báo cáo & giao diện Streamlit
CLASS_LABELS_VI: Dict[str, str] = {
    "combination": "Da hỗn hợp (Combination Skin)",
    "dry": "Da khô (Dry Skin)",
    "normal": "Da thường (Normal Skin)",
    "oily": "Da dầu / nhờn (Oily Skin)",
}

# Bảng màu đại diện trực quan hóa cho từng loại da
CLASS_COLORS: Dict[str, str] = {
    "combination": "#9467BD",  # Tím thanh lịch
    "dry": "#E69F00",          # Cam đất / vàng nhạt
    "normal": "#009E73",       # Xanh lá dịu nhẹ
    "oily": "#0072B2",         # Xanh dương đậm / bóng dầu
}

# ==============================================================================
# 3. THAM SỐ XỬ LÝ ẢNH & TIỀN XỬ LÝ (IMAGE PROCESSING PARAMS)
# ==============================================================================
# Kích thước ảnh chuẩn hóa đầu vào cho toàn bộ pipeline trích xuất
TARGET_SIZE: Tuple[int, int] = (224, 224)
TARGET_IMAGE_SIZE: int = 224

# Các định dạng ảnh được hỗ trợ hợp lệ
VALID_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
)

# Danh sách không gian màu sử dụng trong nghiên cứu
COLOR_SPACES: List[str] = ["RGB", "HSV", "YCbCr", "LAB", "GRAY"]

# Ngưỡng phát hiện ảnh chất lượng kém trong bước thám hiểm dữ liệu
QUALITY_THRESHOLDS: Dict[str, float] = {
    # Ngưỡng phương sai toán tử Laplacian phát hiện ảnh mờ (Blurry threshold)
    # Variance < 100.0 thường coi là ảnh bị mất nét/mờ
    "blur_laplacian_var": 100.0,
    # Ngưỡng độ sáng trung bình (Grayscale mean intensity 0-255)
    "dark_intensity": 50.0,      # Dưới ngưỡng này coi là quá tối
    "bright_intensity": 205.0,   # Trên ngưỡng này coi là quá cháy sáng
    # Ngưỡng kích thước tối thiểu (pixels)
    "min_dimension": 64.0,
}

# ==============================================================================
# 4. NGƯỠNG PHÂN VÙNG DA (SKIN SEGMENTATION THRESHOLDS)
# ==============================================================================
# Phân vùng sắc da trong không gian màu HSV (Hue: 0-179, Saturation: 0-255, Value: 0-255 trong OpenCV)
# Theo nghiên cứu chuẩn thị giác máy tính cho làn da người:
#   Hue: Vùng sắc đỏ-vàng (0 -> 50)
#   Saturation: Độ bão hòa màu da vừa phải (0.2 -> 0.68 tương ứng ~50 -> 175)
#   Value: Cường độ sáng từ trung bình trở lên (0.35 -> 1.0 tương ứng ~80 -> 255)
HSV_SKIN_LOWER: np.ndarray = np.array([0, 50, 80], dtype=np.uint8)
HSV_SKIN_UPPER: np.ndarray = np.array([50, 175, 255], dtype=np.uint8)

# Phân vùng sắc da trong không gian màu YCbCr (Chai & Ngan Rule)
# Y: Cường độ sáng (Luminance) [80, 255]
# Cb: Hiệu màu lam (Chrominance-blue) [77, 127]
# Cr: Hiệu màu đỏ (Chrominance-red) [133, 173]
YCBCR_SKIN_LOWER: np.ndarray = np.array([80, 77, 133], dtype=np.uint8)
YCBCR_SKIN_UPPER: np.ndarray = np.array([255, 127, 173], dtype=np.uint8)

# Ngưỡng lọc điểm chói sáng (Specular Highlights - vùng phản xạ dầu nhờn trên da)
# Vùng dầu nhờn thường có Value rất cao kết hợp Saturation rất thấp trong HSV
HIGHLIGHT_VALUE_THRESHOLD: int = 230
HIGHLIGHT_SAT_THRESHOLD: int = 35

# ==============================================================================
# 5. THAM SỐ TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTION PARAMS)
# ==============================================================================
# Tham số LBP (Local Binary Pattern) trích xuất vi cấu trúc bề mặt da
LBP_RADIUS: int = 1
LBP_N_POINTS: int = 8
LBP_METHOD: str = "uniform"  # Uniform LBP cho tính bất biến quay và gọn nhẹ

# Tham số lược đồ màu (Color Histogram)
HIST_BINS: int = 32
HIST_RANGE: Tuple[int, int] = (0, 256)

# Tham số GLCM (Gray-Level Co-occurrence Matrix)
GLCM_DISTANCES: List[int] = [1, 2, 3]
GLCM_ANGLES: List[float] = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPERTIES: List[str] = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
]

# Các bậc Color Moments (Mean=1, Std=2, Skewness=3)
COLOR_MOMENTS_ORDERS: List[int] = [1, 2, 3]

# ==============================================================================
# 6. THAM SỐ HUẤN LUYỆN & MÔ HÌNH HỌC MÁY (MACHINE LEARNING PARAMS)
# ==============================================================================
RANDOM_STATE: int = 42
SEED: int = RANDOM_STATE

# Tỉ lệ phân chia dữ liệu mặc định (nếu dataset chưa chia sẵn)
DEFAULT_SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)  # Train / Val / Test

# Không gian tham số K-Nearest Neighbors (KNN)
KNN_K_RANGE: List[int] = list(range(1, 22, 2))  # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
KNN_WEIGHTS: List[str] = ["uniform", "distance"]
KNN_METRICS: List[str] = ["euclidean", "manhattan"]

# Không gian tham số Support Vector Machine (SVM)
SVM_KERNELS: List[str] = ["linear", "rbf", "poly"]
SVM_C_RANGE: List[float] = [0.1, 1.0, 10.0, 100.0]
SVM_GAMMA_OPTIONS: List[str] = ["scale", "auto"]

# Tham số Random Forest Baseline
RF_N_ESTIMATORS: List[int] = [50, 100, 200]
RF_MAX_DEPTH: List[int] = [10, 20, 30]

# ==============================================================================
# 7. HÀM TIỆN ÍCH QUẢN LÝ THƯ MỤC (DIRECTORY UTILITIES)
# ==============================================================================
def ensure_directories() -> None:
    """
    Tạo tự động tất cả các thư mục cần thiết trong dự án nếu chưa tồn tại.
    Ensures all necessary directories exist in the project workspace.
    """
    directories = [
        DATA_DIR,
        TRAIN_DIR,
        VALID_DIR,
        TEST_DIR,
        OUTPUTS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        PROCESSED_DATA_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def update_data_dir(new_data_dir: Path | str) -> None:
    """
    Cập nhật lại đường dẫn DATA_DIR và các thư mục train/valid/test tương ứng.
    Dynamically updates DATA_DIR and split directories.
    
    Args:
        new_data_dir (Path | str): Đường dẫn thư mục dữ liệu mới.
    """
    global DATA_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR
    DATA_DIR = Path(new_data_dir).resolve()
    TRAIN_DIR = DATA_DIR / "train"
    VALID_DIR = DATA_DIR / "valid"
    TEST_DIR = DATA_DIR / "test"
