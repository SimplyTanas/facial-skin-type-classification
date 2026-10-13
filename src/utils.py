"""
Module Tiện Ích Chung Cho Dự Án Xử Lý Ảnh Số (Common Utility Functions)
Dự án: Phân loại loại da mặt (Xử lý ảnh số - Digital Image Processing)

File: src/utils.py
Mục đích:
- Đo lường và theo dõi thời gian thực thi (Timer context manager & duration formatting).
- Quản lý lưu trữ và nạp dữ liệu cấu trúc JSON an toàn (hỗ trợ kiểu dữ liệu NumPy/Path).
- Tạo và xác thực hệ thống thư mục xuất dữ liệu (outputs directories).
- Trực quan hóa danh sách ảnh theo dạng lưới ma trận (Grid Display với Matplotlib).
- Thiết lập hệ thống ghi nhật ký hoạt động (Logging setup).
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# 1. BỘ ĐO THỜI GIAN VÀ ĐỊNH DẠNG (TIMER CONTEXT MANAGER & FORMATTING)
# ==============================================================================

class Timer:
    """
    Context manager và bộ đếm thời gian thực thi (Performance Timer).
    Dùng để đo lường độ trễ (latency) từng công đoạn trong xử lý ảnh.

    Cách sử dụng:
        with Timer("Tiền xử lý ảnh") as t:
            # mã nguồn cần đo
            pass
        print(f"Thời gian: {t.elapsed_ms:.2f} ms")

        # Hoặc dùng thủ công:
        timer = Timer().start()
        ...
        timer.stop()
        print(timer)
    """

    def __init__(self, name: Optional[str] = None, verbose: bool = False) -> None:
        """
        Khởi tạo Timer.

        Args:
            name: Tên của công đoạn đo (tùy chọn).
            verbose: Nếu True, tự động in thời gian sau khi kết thúc context block.
        """
        self.name: Optional[str] = name
        self.verbose: bool = verbose
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0
        self._is_running: bool = False

    def start(self) -> Timer:
        """Bắt đầu tính giờ."""
        self.start_time = time.perf_counter()
        self.end_time = 0.0
        self.elapsed = 0.0
        self._is_running = True
        return self

    def stop(self) -> float:
        """
        Dừng tính giờ và trả về số giây trôi qua.

        Returns:
            float: Thời gian thực thi tính theo giây (seconds).
        """
        if self._is_running:
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time
            self._is_running = False
        return self.elapsed

    @property
    def elapsed_ms(self) -> float:
        """Thời gian trôi qua tính theo mili-giây (milliseconds)."""
        if self._is_running:
            return (time.perf_counter() - self.start_time) * 1000.0
        return self.elapsed * 1000.0

    @property
    def formatted(self) -> str:
        """Chuỗi thời gian đã được định dạng thân thiện."""
        return format_duration(self.elapsed if not self._is_running else (time.perf_counter() - self.start_time))

    def __enter__(self) -> Timer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
        if self.verbose:
            prefix = f"[{self.name}] " if self.name else ""
            print(f"{prefix}Hoàn thành trong: {self.formatted}")

    def __repr__(self) -> str:
        name_str = f"'{self.name}': " if self.name else ""
        return f"<Timer {name_str}{self.formatted}>"


def format_duration(seconds: float) -> str:
    """
    Định dạng thời gian tính bằng giây thành chuỗi văn bản thân thiện với con người.
    Format duration into human-readable string (ns, us, ms, s, m s).

    Args:
        seconds: Thời gian tính theo giây (seconds).

    Returns:
        str: Chuỗi văn bản hiển thị thời gian, ví dụ "14.25 ms", "2.10 s", "1m 30.5s".
    """
    if seconds < 0:
        return f"-{format_duration(-seconds)}"
    if seconds == 0:
        return "0.00 ms"

    if seconds < 1e-4:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60.0:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        remaining_sec = seconds % 60
        return f"{minutes}m {remaining_sec:.1f}s"


# ==============================================================================
# 2. XỬ LÝ LƯU TRỮ VÀ ĐỌC JSON (JSON SERIALIZATION WITH NUMPY SUPPORT)
# ==============================================================================

class NumpyJSONEncoder(json.JSONEncoder):
    """
    JSON Encoder mở rộng, tự động chuyển đổi các kiểu dữ liệu NumPy và Path
    thành kiểu dữ liệu chuẩn của Python để serialize an toàn.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


def save_json(
    data: Any,
    filepath: Union[str, Path],
    indent: int = 4,
    ensure_ascii: bool = False
) -> None:
    """
    Lưu dữ liệu cấu trúc từ điển/danh sách vào tệp JSON an toàn (hỗ trợ tiếng Việt UTF-8).
    Save data structure to JSON file with full NumPy and Unicode support.

    Args:
        data: Dữ liệu cần lưu (dict, list, đối tượng có to_dict).
        filepath: Đường dẫn tệp tin đích.
        indent: Độ lùi dòng hiển thị (mặc định 4).
        ensure_ascii: Mặc định False để lưu chính xác ký tự tiếng Việt.
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=indent, ensure_ascii=ensure_ascii)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Đọc dữ liệu từ tệp JSON.
    Load data structure from JSON file.

    Args:
        filepath: Đường dẫn tới tệp JSON cần đọc.

    Returns:
        Any: Dữ liệu đối tượng Python sau khi giải mã.
    """
    path_obj = Path(filepath)
    if not path_obj.is_file():
        raise FileNotFoundError(f"[Lỗi] Không tìm thấy tệp JSON tại: {path_obj}")

    with open(path_obj, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# 3. QUẢN LÝ THƯ MỤC XUẤT DỮ LIỆU (DIRECTORY MANAGEMENT)
# ==============================================================================

def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Tạo thư mục nếu chưa tồn tại và trả về đối tượng Path tuyệt đối.

    Args:
        dir_path: Đường dẫn thư mục cần kiểm tra hoặc tạo mới.

    Returns:
        Path: Đối tượng Path tuyệt đối của thư mục.
    """
    path_obj = Path(dir_path).resolve()
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def ensure_output_dirs(base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """
    Tạo tự động toàn bộ cây thư mục đầu ra tiêu chuẩn cho dự án.
    Creates and returns standard outputs directory hierarchy.

    Args:
        base_dir: Thư mục outputs gốc (mặc định lấy theo PROJECT_ROOT / 'outputs').

    Returns:
        Dict[str, Path]: Bản đồ các đường dẫn thư mục con đã sẵn sàng ghi.
    """
    if base_dir is None:
        # Tự suy luận từ vị trí file hiện tại
        project_root = Path(__file__).resolve().parent.parent
        outputs_root = project_root / "outputs"
    else:
        outputs_root = Path(base_dir).resolve()

    dirs = {
        "outputs": outputs_root,
        "figures": outputs_root / "figures",
        "reports": outputs_root / "reports",
        "models": outputs_root / "models",
        "processed_data": outputs_root / "processed_data",
        "logs": outputs_root / "logs",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


# ==============================================================================
# 4. HIỂN THỊ DANH SÁCH ẢNH DẠNG LƯỚI (IMAGE GRID DISPLAY)
# ==============================================================================

def plot_image_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    suptitle: Optional[str] = None,
    as_rgb: bool = True
) -> plt.Figure:
    """
    Hiển thị danh sách ảnh trên lưới (grid layout) sử dụng Matplotlib.
    Display a list of images in a flexible grid layout with optional titles.

    Args:
        images: Danh sách các mảng ảnh numpy ndarray [H, W, C] hoặc [H, W].
        titles: Danh sách tiêu đề cho từng ô ảnh tương ứng (tùy chọn).
        rows: Số dòng của lưới. Nếu None, tự động tính toán dựa trên cols.
        cols: Số cột của lưới. Nếu None, tự động tính toán hợp lý.
        figsize: Kích thước khung hình đồ họa (chiều rộng, chiều cao).
        cmap: Bảng màu Matplotlib cho ảnh đơn kênh (ví dụ 'gray', 'viridis').
        save_path: Đường dẫn để lưu biểu đồ ra đĩa (tùy chọn).
        suptitle: Tiêu đề tổng quát ở đỉnh đồ thị (tùy chọn).
        as_rgb: Nếu ảnh có 3 kênh và người dùng đưa vào là RGB thì hiển thị trực tiếp.

    Returns:
        plt.Figure: Đối tượng figure của Matplotlib.
    """
    n_images = len(images)
    if n_images == 0:
        raise ValueError("[Lỗi] Danh sách ảnh rỗng, không thể hiển thị lưới.")

    # Tự động tính số hàng và cột nếu chưa chỉ định
    if rows is None and cols is None:
        cols = min(4, n_images)
        rows = math.ceil(n_images / cols)
    elif cols is None:
        cols = math.ceil(n_images / rows)
    elif rows is None:
        rows = math.ceil(n_images / cols)

    if figsize is None:
        figsize = (cols * 3.5, rows * 3.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Chuyển đổi axes thành mảng 1 chiều tiện duyệt
    if isinstance(axes, np.ndarray):
        flat_axes = axes.flatten()
    else:
        flat_axes = [axes]

    for idx in range(len(flat_axes)):
        ax = flat_axes[idx]
        if idx < n_images:
            img = images[idx]
            # Xử lý hiển thị ảnh
            if img.ndim == 2:
                ax.imshow(img, cmap=cmap if cmap else "gray")
            elif img.ndim == 3:
                # Nếu ảnh dạng float trong dải [0, 1] hoặc uint8 [0, 255]
                ax.imshow(img)
            else:
                ax.imshow(img.squeeze(), cmap=cmap if cmap else "gray")

            if titles and idx < len(titles):
                ax.set_title(str(titles[idx]), fontsize=10, fontweight="bold")
            ax.axis("off")
        else:
            # Ẩn các ô trống thừa
            ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200, bbox_inches="tight")

    return fig


# ==============================================================================
# 5. THIẾT LẬP GHI NHẬT KÝ (LOGGING SETUP)
# ==============================================================================

def setup_logger(
    name: str = "xla_pipeline",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Khởi tạo hoặc lấy logger cấu hình chuẩn cho quy trình xử lý ảnh.
    Configures and returns a structured logger with console and optional file output.

    Args:
        name: Tên của logger (thường là tên module hoặc 'xla_pipeline').
        log_file: Đường dẫn tệp ghi log (tùy chọn).
        level: Cấp độ ghi nhận (mặc định logging.INFO).

    Returns:
        logging.Logger: Đối tượng logger đã cấu hình xong formatter và handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Tránh gắn handler nhiều lần nếu hàm được gọi lặp lại
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (nếu được yêu cầu)
    if log_file:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
