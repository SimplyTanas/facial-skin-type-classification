"""
Digital Image Processing (Xử lý ảnh số) - Skin Type Classification Project
Phase 2: Image Processing Pipeline & Feature Engineering

Modules:
- config: Tham số cấu hình dự án, ngưỡng màu, đường dẫn.
- preprocessing: Tiền xử lý ảnh (đọc, kiểm định, resize, khử nhiễu, CLAHE, baseline/enhanced pipelines).
- skin_detection: Phân đoạn vùng da, lọc hình thái học, trích xuất ROI, đánh giá chất lượng mặt nạ.
- color_space: Chuyển đổi và phân tích các không gian màu (RGB, HSV, YCrCb, LAB, Grayscale).
- feature_extraction: Trích xuất đặc trưng màu sắc, độ sáng, và kết cấu LBP trên vùng da.
- feature_normalization: Chuẩn hóa đặc trưng (StandardScaler) và giảm chiều PCA.
- pipeline: Đường ống xử lý hoàn chỉnh (BaselinePipeline, EnhancedPipeline, Batch Processing).
- utils: Tiện ích đo thời gian, JSON serialization, lưu trữ thư mục, trực quan hóa lưới ảnh.
"""

from src import (
    color_space,
    config,
    feature_extraction,
    feature_normalization,
    pipeline,
    preprocessing,
    skin_detection,
    utils,
)

__all__ = [
    "config",
    "preprocessing",
    "skin_detection",
    "color_space",
    "feature_extraction",
    "feature_normalization",
    "pipeline",
    "utils",
]
