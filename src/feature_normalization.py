"""
Module Chuẩn Hóa và Giảm Chiều Không Gian Đặc Trưng (Feature Normalization & PCA Module)
Dự án: Phân loại loại da mặt (Xử lý ảnh số - Digital Image Processing)

File: src/feature_normalization.py
Mục đích:
- Chuẩn hóa phân phối đặc trưng bằng StandardScaler (Z-score normalization: mean=0, std=1).
- Tùy chọn giảm số chiều không gian đặc trưng bằng PCA (Principal Component Analysis).
- Ngăn ngừa hiện tượng rò rỉ thông tin dữ liệu (Data Leakage):
  + CHỈ fit (học tham số mean, std, thành phần chính) trên tập huấn luyện (Train set).
  + Áp dụng transform (biến đổi) đồng nhất lên tập Validation, Test và ảnh suy luận thực tế.
- Đóng gói toàn bộ quy trình thành lớp `FeatureNormalizer` tiện lợi và tái sử dụng cao.
- Lưu trữ và phục hồi bộ chuẩn hóa bằng joblib để triển khai mô hình sản phẩm.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Nhập cấu hình nếu có
try:
    from src import config
    DEFAULT_RANDOM_STATE = getattr(config, "RANDOM_STATE", 42)
except ImportError:
    DEFAULT_RANDOM_STATE = 42

logger = logging.getLogger("xla_pipeline.normalization")


# ==============================================================================
# 1. CÁC HÀM XỬ LÝ CHUẨN HÓA THUẦN TÚY (FUNCTIONAL SCALER API)
# ==============================================================================

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Huấn luyện bộ chuẩn hóa StandardScaler CHỈ trên tập Train.
    Fit StandardScaler on training features only (avoids data leakage).

    Quy tắc:
        z = (x - u) / s
        với u là trung bình mẫu train, s là độ lệch chuẩn mẫu train.

    Args:
        X_train: Ma trận đặc trưng tập huấn luyện [N_samples, N_features].

    Returns:
        StandardScaler: Đối tượng scaler đã học xong tham số mean_ và scale_.

    Raises:
        ValueError: Nếu mảng đầu vào không phải 2D hoặc chứa giá trị NaN/Inf.
    """
    if not isinstance(X_train, np.ndarray) or X_train.ndim != 2:
        raise ValueError(f"[Lỗi] X_train phải là mảng 2 chiều [N_samples, N_features]. Nhận được: {type(X_train)}")
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        raise ValueError("[Lỗi] X_train chứa giá trị bất thường (NaN hoặc Inf). Cần xử lý trước khi fit.")

    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info(f"Đã huấn luyện StandardScaler trên {X_train.shape[0]} mẫu, {X_train.shape[1]} đặc trưng.")
    return scaler


def transform_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Biến đổi chuẩn hóa ma trận đặc trưng bằng scaler đã huấn luyện trước đó.
    Transform feature matrix using pre-fitted StandardScaler.

    Args:
        X: Ma trận đặc trưng [N_samples, N_features].
        scaler: Đối tượng StandardScaler đã được fit.

    Returns:
        np.ndarray: Ma trận đặc trưng chuẩn hóa dạng float32.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError(f"[Lỗi] X phải là mảng 2 chiều [N_samples, N_features]. Nhận được: {type(X)}")
    if not hasattr(scaler, "mean_") or scaler.mean_ is None:
        raise ValueError("[Lỗi] Scaler chưa được huấn luyện (chưa gọi fit).")
    if X.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"[Lỗi] Số chiều đặc trưng không khớp: X có {X.shape[1]} cột, "
            f"scaler kỳ vọng {scaler.n_features_in_} cột."
        )

    # Thay thế tạm thời các giá trị NaN/Inf nếu có phát sinh đột xuất
    X_clean = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    X_scaled = scaler.transform(X_clean)
    return X_scaled.astype(np.float32)


# ==============================================================================
# 2. CÁC HÀM XỬ LÝ GIẢM CHIỀU PCA (FUNCTIONAL PCA API)
# ==============================================================================

def fit_pca(
    X_train_scaled: np.ndarray,
    n_components: Union[int, float] = 0.95,
    random_state: int = DEFAULT_RANDOM_STATE
) -> PCA:
    """
    Huấn luyện thuật toán phân tích thành phần chính (PCA) trên dữ liệu ĐÃ chuẩn hóa của tập Train.
    Fit Principal Component Analysis (PCA) on scaled training data.

    Args:
        X_train_scaled: Ma trận đặc trưng tập train đã qua StandardScaler [N, D].
        n_components:
            - Nếu là số thực (0.0 < float < 1.0): Tỷ lệ phương sai tích lũy cần bảo toàn
              (ví dụ 0.95 giữ lại 95% thông tin biến thiên của dữ liệu).
            - Nếu là số nguyên (int >= 1): Số lượng thành phần chính cố định cần trích xuất.
        random_state: Hạt giống ngẫu nhiên đảm bảo tính tái lập.

    Returns:
        PCA: Đối tượng PCA đã học xong ma trận vector riêng và phương sai giải thích.
    """
    if not isinstance(X_train_scaled, np.ndarray) or X_train_scaled.ndim != 2:
        raise ValueError("[Lỗi] X_train_scaled phải là mảng 2 chiều.")

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_train_scaled)

    total_var = float(np.sum(pca.explained_variance_ratio_))
    logger.info(
        f"Đã huấn luyện PCA: từ {X_train_scaled.shape[1]} chiều -> {pca.n_components_} thành phần chính "
        f"(giải thích {total_var * 100:.2f}% tổng phương sai)."
    )
    return pca


def transform_pca(X_scaled: np.ndarray, pca: PCA) -> np.ndarray:
    """
    Chiếu ma trận đặc trưng đã chuẩn hóa sang không gian các thành phần chính PCA.
    Project scaled feature matrix into PCA subspace.

    Args:
        X_scaled: Ma trận đặc trưng đã chuẩn hóa [N_samples, N_features].
        pca: Đối tượng PCA đã fit.

    Returns:
        np.ndarray: Ma trận đặc trưng sau khi giảm chiều [N_samples, n_components].
    """
    if not isinstance(X_scaled, np.ndarray) or X_scaled.ndim != 2:
        raise ValueError("[Lỗi] X_scaled phải là mảng 2 chiều.")
    if not hasattr(pca, "components_"):
        raise ValueError("[Lỗi] Đối tượng PCA chưa được fit.")
    if X_scaled.shape[1] != pca.n_features_in_:
        raise ValueError(
            f"[Lỗi] Số chiều không khớp: X_scaled có {X_scaled.shape[1]} cột, "
            f"PCA yêu cầu {pca.n_features_in_} cột."
        )

    X_proj = pca.transform(X_scaled)
    return X_proj.astype(np.float32)


# ==============================================================================
# 3. LỚP ĐÓNG GÓI CHUẨN HÓA VÀ GIẢM CHIỀU TOÀN DIỆN (FEATURE NORMALIZER CLASS)
# ==============================================================================

class FeatureNormalizer:
    """
    Lớp đóng gói trọn vẹn quy trình Chuẩn hóa (StandardScaler) và Giảm chiều (PCA).
    Unified Feature Normalizer wrapping StandardScaler and optional PCA.

    NGUYÊN TẮC QUAN TRỌNG:
    - Phương thức `fit` hoặc `fit_transform` CHỈ ĐƯỢC PHÉP GỌI DUY NHẤT TRÊN TẬP TRAIN.
    - Phương thức `transform` dùng chung cho Train, Validation, Test và các ảnh mới.
    - Điều này đảm bảo tính khách quan tuyệt đối của bài toán học máy trong xử lý ảnh số.

    Ví dụ:
        normalizer = FeatureNormalizer(use_pca=True, n_components=0.95)
        X_train_norm = normalizer.fit_transform(X_train, feature_names=names)
        X_val_norm = normalizer.transform(X_val)
        X_test_norm = normalizer.transform(X_test)

        # Lưu lại để deploy:
        normalizer.save("outputs/models/feature_normalizer.joblib")
    """

    def __init__(
        self,
        use_pca: bool = False,
        n_components: Optional[Union[int, float]] = 0.95,
        random_state: int = DEFAULT_RANDOM_STATE
    ) -> None:
        """
        Khởi tạo FeatureNormalizer.

        Args:
            use_pca: Có áp dụng PCA giảm chiều sau khi chuẩn hóa hay không.
            n_components: Tham số số chiều hoặc tỉ lệ phương sai tích lũy của PCA.
            random_state: Hạt giống ngẫu nhiên.
        """
        self.use_pca: bool = use_pca
        self.n_components: Optional[Union[int, float]] = n_components
        self.random_state: int = random_state

        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.n_features_in_: int = 0
        self.n_features_out_: int = 0
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> FeatureNormalizer:
        """
        Huấn luyện bộ chuẩn hóa trên tập Train.
        Fit scaler and optional PCA on training data only.

        Args:
            X_train: Ma trận đặc trưng tập Train [N, D].
            feature_names: Danh sách tên các đặc trưng đầu vào (tùy chọn).

        Returns:
            FeatureNormalizer: Đối tượng chính nó đã được fit xong.
        """
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2:
            raise ValueError("[Lỗi] X_train cần có dạng ma trận 2 chiều [N_samples, N_features].")

        self.n_features_in_ = X_train.shape[1]
        self.feature_names_in_ = list(feature_names) if feature_names else None

        # 1. Fit StandardScaler
        self.scaler = fit_scaler(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        # 2. Fit PCA (nếu kích hoạt)
        if self.use_pca:
            # Điều chỉnh n_components an toàn nếu số mẫu ít hơn số chiều
            max_possible = min(X_train.shape[0], X_train.shape[1])
            n_comp = self.n_components
            if isinstance(n_comp, int) and n_comp > max_possible:
                logger.warning(
                    f"n_components ({n_comp}) > min(samples, features) = {max_possible}. "
                    f"Tự động điều chỉnh về {max_possible}."
                )
                n_comp = max_possible

            self.pca = fit_pca(
                X_train_scaled,
                n_components=n_comp if n_comp is not None else 0.95,
                random_state=self.random_state
            )
            self.n_features_out_ = int(self.pca.n_components_)
        else:
            self.pca = None
            self.n_features_out_ = self.n_features_in_

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Biến đổi ma trận đặc trưng sử dụng các tham số đã fit.
        Transform feature matrix using fitted scaler and PCA.

        Args:
            X: Ma trận đặc trưng [N, D] (có thể là train, val, test hoặc 1 mẫu [1, D]).

        Returns:
            np.ndarray: Ma trận sau khi chuẩn hóa và giảm chiều.
        """
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("[Lỗi] FeatureNormalizer chưa được huấn luyện. Hãy gọi fit() trước.")

        # Xử lý vector 1D thành 2D nếu người dùng đưa vào 1 mẫu đơn lẻ
        if X.ndim == 1:
            X_2d = X.reshape(1, -1)
            is_single = True
        else:
            X_2d = X
            is_single = False

        # 1. Chuẩn hóa qua scaler
        X_scaled = transform_scaler(X_2d, self.scaler)

        # 2. Chiếu qua PCA nếu có
        if self.use_pca and self.pca is not None:
            X_out = transform_pca(X_scaled, self.pca)
        else:
            X_out = X_scaled

        if is_single:
            return X_out.squeeze(0)
        return X_out

    def fit_transform(
        self,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Thực hiện cả fit và transform đồng thời trên tập Train.
        Fit and transform on training data.
        """
        self.fit(X_train, feature_names=feature_names)
        return self.transform(X_train)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Biến đổi ngược lại miền giá trị đặc trưng ban đầu (phục vụ giải thích).
        Inverse transform back to original feature scale.
        """
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("[Lỗi] Normalizer chưa được fit.")

        if self.use_pca and self.pca is not None:
            X_scaled = self.pca.inverse_transform(X_transformed)
        else:
            X_scaled = X_transformed

        return self.scaler.inverse_transform(X_scaled)

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Trả về mảng tỷ lệ phương sai giải thích của các thành phần PCA (nếu có)."""
        if self.use_pca and self.pca is not None:
            return self.pca.explained_variance_ratio_
        return None

    def summary(self) -> Dict[str, Any]:
        """Tổng hợp thông tin cấu hình và trạng thái của bộ chuẩn hóa."""
        res = {
            "is_fitted": self.is_fitted,
            "use_pca": self.use_pca,
            "n_features_in": self.n_features_in_,
            "n_features_out": self.n_features_out_,
            "feature_names_in": self.feature_names_in_,
        }
        if self.use_pca and self.pca is not None:
            res["pca_n_components"] = int(self.pca.n_components_)
            res["pca_cumulative_variance"] = float(np.sum(self.pca.explained_variance_ratio_))
        return res

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Lưu trạng thái bộ chuẩn hóa vào tệp tin đĩa (joblib).
        Save normalizer instance to disk.
        """
        save_normalizer(filepath, self)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> FeatureNormalizer:
        """
        Nạp bộ chuẩn hóa từ tệp tin đĩa.
        Load normalizer instance from disk.
        """
        return load_normalizer(filepath)


# ==============================================================================
# 4. LƯU VÀ NẠP BỘ CHUẨN HÓA QUA JOBLIB (DISK PERSISTENCE API)
# ==============================================================================

def save_normalizer(
    filepath: Union[str, Path],
    normalizer: Any
) -> None:
    """
    Lưu trữ đối tượng chuẩn hóa (FeatureNormalizer hoặc dict/scaler) ra đĩa.
    Save normalizer object to disk using joblib.

    Args:
        filepath: Đường dẫn lưu tệp (ví dụ .joblib hoặc .pkl).
        normalizer: Đối tượng cần lưu.
    """
    out_path = Path(filepath).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(normalizer, out_path)
    logger.info(f"Đã lưu thành công Feature Normalizer tại: {out_path}")


def load_normalizer(filepath: Union[str, Path]) -> Any:
    """
    Nạp đối tượng chuẩn hóa từ tệp tin đĩa.
    Load normalizer object from disk using joblib.

    Args:
        filepath: Đường dẫn tệp cần nạp.

    Returns:
        Any: Đối tượng FeatureNormalizer hoặc scaler đã lưu.
    """
    in_path = Path(filepath).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"[Lỗi] Không tìm thấy file normalizer tại: {in_path}")

    loaded_obj = joblib.load(in_path)
    logger.info(f"Đã nạp thành công Feature Normalizer từ: {in_path}")
    return loaded_obj
