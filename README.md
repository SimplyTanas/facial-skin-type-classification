# ĐỒ ÁN XỬ LÝ ẢNH SỐ: PHÂN LOẠI LOẠI DA MẶT (FACIAL SKIN TYPE CLASSIFICATION)

> **Đề tài**: Phân loại loại da mặt (*Dry, Normal, Oily, Combination Skin Types*) sử dụng Kỹ thuật Xử lý ảnh số (Digital Image Processing) & Học máy (Machine Learning).

---

## 1. GIỚI THIỆU DỰ ÁN

Dự án nghiên cứu và xây dựng hệ thống tự động nhận diện và phân loại loại da mặt người dựa trên các đặc tính quang học bề mặt, phân bố màu sắc, phản xạ ánh sáng (bóng dầu) và kết cấu biểu bì vi mô (lỗ chân lông, nếp nhăn). 

Hệ thống kết hợp giữa các thuật toán **Xử lý ảnh số truyền thống** (tiền xử lý khử nhiễu, cân bằng sáng thích ứng CLAHE, phân vùng da đa không gian màu, lọc hình thái học, trích xuất đặc trưng LBP & Color Moments) và các mô hình **Học máy kinh điển** (KNN, SVM), mang lại tính tường minh cao và khả năng giải thích toán học rõ ràng.

### Các loại da mục tiêu:
- **Da khô (Dry Skin)**: Bề mặt lỳ (matte), phản xạ khuếch tán Lambertian, thô ráp, nếp nhăn vi mô, lỗ chân lông khép kín.
- **Da thường (Normal Skin)**: Cân bằng nước - dầu lý tưởng, bề mặt mịn màng đồng nhất, phản xạ sáng mềm mại.
- **Da dầu (Oily Skin)**: Tuyến bã nhờn hoạt động mạnh, xuất hiện các điểm phản xạ gương chói lóa (Specular Highlights), lỗ chân lông giãn nở to dạng "vỏ quả cam".
- **Da hỗn hợp (Combination Skin)**: Vùng chữ T (trán, mũi, cằm) bóng dầu trong khi hai bên má khô hoặc bình thường.

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
XLA/
├── data/                                # Thư mục chứa bộ dữ liệu (hoặc skin_type_classification_dataset/)
│   ├── train/                           # Phân tập huấn luyện (combination, dry, normal, oily)
│   ├── valid/                           # Phân tập kiểm định (combination, dry, normal, oily)
│   └── test/                            # Phân tập kiểm thử độc lập (combination, dry, normal, oily)
├── docs/                                # Tài liệu học thuật và báo cáo đồ án
│   ├── chuong3_khao_sat_du_lieu.md      # Báo cáo Chương 3 chi tiết dạng Markdown
│   ├── chuong3_khao_sat_du_lieu.docx    # Báo cáo Chương 3 xuất bản dạng Word (.docx) chuẩn học thuật
│   └── generate_chapter3_docx.py        # Kịch bản tự động chuyển đổi Markdown sang Word
├── outputs/                             # Thư mục lưu kết quả tự động sinh ra
│   ├── figures/                         # Các biểu đồ trực quan hóa (.png)
│   ├── models/                          # Các mô hình đã lưu (.joblib)
│   ├── processed_data/                  # Ma trận đặc trưng đã trích xuất (.npz, .csv)
│   ├── reports/                         # Báo cáo định lượng và danh mục ảnh cảnh báo (.json, .csv)
│   ├── dataset_metadata.csv             # Bảng siêu dữ liệu toàn bộ ảnh
│   └── phase1_dataset_summary.json      # Tệp JSON tổng hợp thống kê Phase 1
├── src/                                 # Mã nguồn các mô-đun lõi
│   ├── __init__.py                      # Khởi tạo package và cấu hình UTF-8
│   ├── color_space.py                   # Chuyển đổi và phân tích không gian màu (RGB, HSV, YCrCb, LAB)
│   ├── config.py                        # Cấu hình tập trung siêu tham số và đường dẫn
│   ├── data_explorer.py                 # Khám phá, trực quan hóa và kiểm định chất lượng ảnh
│   ├── data_loader.py                   # Tải dữ liệu, chia tập phân tầng, băm MD5/dHash
│   ├── feature_extraction.py            # Trích xuất 126 đặc trưng (Màu sắc, Độ bóng sáng, LBP)
│   ├── feature_normalization.py         # Chuẩn hóa Z-Score và giảm chiều PCA
│   ├── pipeline.py                      # BaselinePipeline & EnhancedPipeline
│   ├── preprocessing.py                 # Tiền xử lý ảnh (Resize, Gaussian Blur, CLAHE)
│   ├── skin_detection.py                # Phân đoạn da, lọc hình thái học, trích xuất ROI
│   └── utils.py                         # Tiện ích đo thời gian, JSON Numpy, Logging
├── tests/                               # Bộ kiểm thử tính năng (Unit Tests)
│   └── test_phase2.py                   # Kiểm thử tích hợp các mô-đun tiền xử lý và trích xuất
├── phase1_data_preparation.py           # Kịch bản thực thi Giai đoạn 1 (Chuẩn bị dữ liệu)
├── phase2_build_pipeline.py             # Kịch bản thực thi Giai đoạn 2 (Xây dựng Pipeline & Đặc trưng)
├── requirements.txt                     # Danh sách các thư viện phụ thuộc
└── README.md                            # Hướng dẫn sử dụng dự án
```

---

## 3. CÀI ĐẶT MÔI TRƯỜNG

### 3.1. Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11, Linux, hoặc macOS.
- **Phiên bản Python**: Python `3.9` trở lên (khuyến nghị `3.10` - `3.13`).

### 3.2. Cài đặt các thư viện cần thiết
Mở terminal (PowerShell, Command Prompt hoặc Bash) và thực hiện:

```bash
# 1. Di chuyển vào thư mục dự án
cd [đường_dẫn_tới_thư_mục_XLA]

# 2. Cài đặt toàn bộ phụ thuộc từ requirements.txt
pip install -r requirements.txt
```

> **Lưu ý trên Windows**: Để đảm bảo hiển thị tiếng Việt và các ký tự đặc biệt (`✓`, `★`) mượt mà trên PowerShell / Command Prompt mà không bị lỗi `UnicodeEncodeError`, bạn có thể chạy lệnh sau trước khi thực thi script:
> ```powershell
> $env:PYTHONUTF8=1
> ```
> *(Trong mã nguồn dự án đã tích hợp sẵn cơ chế tự động reconfigure UTF-8 ở đầu các file script).*

---

## 4. CHUẨN BỊ DỮ LIỆU (DATASET)

Dự án sử dụng bộ dữ liệu **"Oily, Dry and Normal Skin Types Dataset"** gồm **4,093 ảnh JPEG** ($640 \times 640$ px, dung lượng 163.63 MB).

1. Tải bộ dữ liệu từ Kaggle hoặc giải nén tệp dữ liệu có sẵn.
2. Đặt thư mục dữ liệu vào thư mục gốc của dự án với tên `data/` hoặc `skin_type_classification_dataset/`.
3. Cấu trúc thư mục dữ liệu chuẩn:
   ```text
   data/
   ├── train/{combination, dry, normal, oily}
   ├── valid/{combination, dry, normal, oily}
   └── test/{combination, dry, normal, oily}
   ```
*(Hệ thống đã tích hợp sẵn cơ chế tự động phát hiện đường dẫn dữ liệu ưu tiên: `skin_type_classification_dataset` -> `data` -> các thư mục ứng viên khác).*

---

## 5. HƯỚNG DẪN CHẠY DỰ ÁN

### 5.1. Chạy Giai đoạn 1: Chuẩn bị & Khảo sát Dữ liệu (Phase 1)
Kịch bản `phase1_data_preparation.py` thực hiện:
- Tự động nhận diện cấu trúc dataset và kiểm tra tính hợp lệ từng tệp ảnh.
- Thống kê phân bố lớp, độ phân giải, tỷ lệ khung hình và dung lượng.
- Bóc tách ảnh trùng lặp tuyệt đối (MD5) và cảm nhận thị giác (dHash).
- Phân tích chất lượng ảnh số (độ mờ Laplacian, độ sáng quá mức/quá tối).
- Xuất bản báo cáo Markdown và 4 biểu đồ trực quan hóa vào thư mục `outputs/`.

```bash
# Chạy với tham số mặc định (tự nhận diện dataset)
python phase1_data_preparation.py

# Hoặc chỉ định đường dẫn dữ liệu tùy biến
python phase1_data_preparation.py --data-dir "skin_type_classification_dataset"

# Giới hạn số mẫu ảnh phân tích chất lượng sâu (mặc định: 500)
python phase1_data_preparation.py --sample-quality-size 500
```

### 5.2. Chạy Giai đoạn 2: Pipeline Xử lý Ảnh & Trích xuất Đặc trưng (Phase 2)
Kịch bản `phase2_build_pipeline.py` thực hiện tuần tự 9 bước logic:
1. Nạp các ảnh mẫu đại diện từng lớp (`dry`, `normal`, `oily`).
2. Demo tiền xử lý cơ bản (Baseline: Resize trực tiếp $224 \times 224$).
3. Demo tiền xử lý nâng cao (Enhanced: Gaussian Denoise + CLAHE trên kênh Luminance L).
4. Phân đoạn vùng da (Skin Detection: kết hợp HSV AND YCrCb + Morphology lọc sạch).
5. So sánh biểu diễn qua 5 không gian màu (RGB, HSV, YCrCb, LAB, Grayscale).
6. Trích xuất **126 chiều đặc trưng** (Nhóm A: Màu sắc 105 chiều, Nhóm B: Độ bóng nhờn 7 chiều, Nhóm C: LBP 14 chiều).
7. Xử lý lô bộ dữ liệu qua cả BaselinePipeline và EnhancedPipeline.
8. Chuẩn hóa đặc trưng Z-Score, giảm chiều PCA còn **13 chiều** (giữ lại $95.16\%$ phương sai) và lưu file `.npz`, `.joblib`.
9. Đánh giá so sánh định lượng độ trễ thời gian (latency ms) và chất lượng mặt nạ da.

```bash
# Chạy demo nhanh 30 mẫu ảnh (mặc định tối ưu để kiểm thử nhanh)
python phase2_build_pipeline.py

# Xử lý số lượng mẫu tùy chọn (ví dụ 100 ảnh)
python phase2_build_pipeline.py --max-samples 100

# Bật/tắt giảm chiều PCA (mặc định: bật với tỷ lệ phương sai 0.95)
python phase2_build_pipeline.py --use-pca --pca-components 0.95
```

### 5.3. Chạy Kiểm thử Đơn vị (Unit Tests)
Chạy bộ kiểm thử tự động để đảm bảo tính đồng bộ của toàn bộ các hàm tiền xử lý, phân đoạn da và không gian màu:

```bash
python tests/test_phase2.py
```

### 5.4. Tự động Xuất Báo cáo Học thuật Word (.docx)
Chuyển đổi báo cáo Chương 3 Markdown sang văn bản Word hoàn chỉnh chuẩn đồ án tốt nghiệp:

```bash
python docs/generate_chapter3_docx.py
```
> Tệp kết quả được lưu tại: [docs/chuong3_khao_sat_du_lieu.docx](file:///c:/Users/baota/Downloads/XLA/docs/chuong3_khao_sat_du_lieu.docx).

---

## 6. KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG

Kết quả đo đạc thực chứng từ hệ thống:

| Chỉ số kỹ thuật | Baseline Pipeline | Enhanced Pipeline | Cải thiện / Ý nghĩa |
| :--- | :---: | :---: | :--- |
| **Thời gian xử lý trung bình/ảnh** | **$60.77$ ms** | **$61.63$ ms** | Chênh lệch chỉ $+0.86$ ms, đáp ứng thời gian thực ($>16$ FPS) |
| **Tỷ lệ diện tích da bóc tách** | **$32.53\%$** | **$36.24\%$** | **Tăng $+3.71\%$ diện tích da** nhờ phối hợp HSV+YCrCb & Morphology |
| **Tỷ lệ mặt nạ da hợp lệ ($>5\%$)** | **$93.33\%$** | **$93.33\%$** | Ổn định cao, lọc bỏ chính xác ảnh không chứa da |
| **Số chiều vector đặc trưng thô** | 126 chiều | 126 chiều | Gồm 105 màu sắc, 7 độ sáng & chói bóng, 14 kết cấu LBP |
| **Giảm chiều đặc trưng (PCA)** | - | **13 chiều** | **Giữ lại $95.16\%$ phương sai**, giảm $89.7\%$ số chiều |

---

## 7. DANH MỤC BIỂU ĐỒ ĐÃ KẾT XUẤT

Các biểu đồ phân tích chất lượng cao (DPI 200 - 300) được lưu trong [outputs/figures/](file:///c:/Users/baota/Downloads/XLA/outputs/figures):
1. `class_distribution.png`: Biểu đồ cân bằng các lớp và phân chia phân tập dữ liệu.
2. `sample_images_grid.png`: Lưới ảnh chân dung khuôn mặt thực tế đại diện từng loại da.
3. `image_size_distribution.png`: Tán xạ độ phân giải ($640 \times 640$) và tỷ lệ khung hình ($1.0$).
4. `image_quality_distribution.png`: Phân bố độ sắc nét Laplacian, độ sáng và vùng chói bóng nhờn.
5. `phase2_step2_baseline_preprocessing.png`: Thử nghiệm resize trực tiếp của Baseline Pipeline.
6. `phase2_step3_enhanced_preprocessing.png`: Khử nhiễu Gaussian và cân bằng sáng thích ứng CLAHE.
7. `phase2_step4_skin_detection_comparison.png`: So sánh phân đoạn da HSV vs YCrCb vs Morphology.
8. `phase2_step5_color_spaces.png`: Biểu diễn qua 5 không gian màu (RGB, HSV, YCrCb, LAB, Grayscale).
9. `phase2_step6_features_breakdown.png`: Phân rã 126 chiều đặc trưng (Nhóm A, Nhóm B, Nhóm C).
10. `phase2_step9_pipeline_comparison.png`: Biểu đồ so sánh định lượng độ trễ và chất lượng mặt nạ da.
11. `phase2_single_image_comparison.png`: Lưới so sánh trực quan $2 \times 4$ trên cùng 1 ảnh giữa Baseline và Enhanced.

---

## 8. TÁC GIẢ & BẢN QUYỀN
- **Đồ án môn học**: Xử lý ảnh số (Digital Image Processing).
- **Hệ điều hành kiểm thử**: Windows 11 / Windows 10, Python 3.13.
- Toàn bộ mã nguồn và tài liệu học thuật được tổ chức mở phục vụ mục đích nghiên cứu và học tập.
