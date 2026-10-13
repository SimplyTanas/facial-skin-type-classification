# CHƯƠNG 3. KHẢO SÁT DỮ LIỆU VÀ PHÂN TÍCH BÀI TOÁN

---

Trong các bài toán thuộc lĩnh vực Xử lý ảnh số (Digital Image Processing) và Thị giác máy tính (Computer Vision), chất lượng và đặc tính của dữ liệu đầu vào đóng vai trò quyết định đến sự thành công của toàn bộ quy trình xây dựng mô hình. Dữ liệu không chỉ cung cấp thông tin để thuật toán học hỏi mà còn phản ánh trực tiếp các thách thức thực tế mà hệ thống phải đối mặt trong môi trường triển khai. Đối với bài toán phân loại loại da mặt người (Facial Skin Type Classification), dữ liệu ảnh đối mặt với nhiều biến số phức tạp như đặc điểm giải phẫu học của khuôn mặt, sự khác biệt về sắc tố da sinh học, điều kiện chiếu sáng phi kiểm soát, góc chụp, cũng như sự can thiệp của mỹ phẩm trang điểm.

Chương này tập trung trình bày một cách toàn diện và có hệ thống về quá trình khảo sát, phân tích định lượng và định tính bộ dữ liệu thực tế **"Oily, Dry and Normal Skin Types Dataset"**, bao gồm xuất xứ, cấu trúc phân cấp, đặc điểm thị giác đặc trưng của từng lớp da, đánh giá hiện tượng mất cân bằng dữ liệu, quy trình làm sạch - kiểm tra tính toàn vẹn (tính hợp lệ, mã băm MD5, dHash, các chỉ số mờ/tối/sáng), cũng như các nguyên tắc chia tách dữ liệu nghiêm ngặt nhằm ngăn chặn rò rỉ thông tin (data leakage). Trên cơ sở các kết quả thực nghiệm thu được từ Giai đoạn 1 (Chuẩn bị dữ liệu) và Giai đoạn 2 (Xây dựng đường ống xử lý ảnh & trích xuất đặc trưng), chương cung cấp các bảng số liệu thực chứng, đánh giá so sánh định lượng giữa đường ống cơ sở (Baseline Pipeline) và đường ống nâng cao (Enhanced Pipeline), tạo tiền đề vững chắc cho việc huấn luyện các mô hình học máy phân loại ở các giai đoạn tiếp theo.

---

## 3.1. GIỚI THIỆU BỘ DỮ LIỆU

### 3.1.1. Xuất xứ và mục tiêu của bộ dữ liệu

Bộ dữ liệu được sử dụng trong đề tài nghiên cứu này có tên gọi chính thức là **"Oily, Dry and Normal Skin Types Dataset"** (thường được biết đến trên các diễn đàn khoa học với tên gọi *Facial Skin Analysis and Type Classification Dataset*), được công bố và chia sẻ mở trên nền tảng khoa học dữ liệu **Kaggle** bởi tác giả **Shakya Dissanayake**.

Mục tiêu cốt lõi của bộ dữ liệu là cung cấp một tập hợp có giám sát (supervised dataset) gồm các hình ảnh chân dung khuôn mặt thực tế phục vụ nghiên cứu và phát triển các giải thuật phân loại loại da mặt người thành các nhóm sinh lý cơ bản:
1. **Da khô (Dry skin)**: Tình trạng da thiếu hụt độ ẩm và lipid bề mặt, có xu hướng thô ráp, nứt nẻ và bong tróc vi mô.
2. **Da thường (Normal skin)**: Trạng thái da lý tưởng với sự cân bằng sinh học hoàn hảo giữa lượng dầu và nước, bề mặt mịn màng và đồng nhất.
3. **Da dầu (Oily skin)**: Tình trạng tuyến bã nhờn hoạt động quá mức, dẫn đến sự tích tụ bã nhờn trên bề mặt da, tạo ra độ bóng nhờn rõ rệt và lỗ chân lông giãn to.
4. **Da hỗn hợp (Combination skin)**: Nhóm da biểu hiện đồng thời cả hai trạng thái: vùng chữ T (trán, mũi, cằm) tiết nhiều dầu nhờn trong khi hai bên má lại ở trạng thái khô ráp hoặc bình thường.

Trong thực tiễn y khoa da liễu và ngành công nghiệp chăm sóc sắc đẹp (BeautyTech), việc phân định chính xác loại da là tiền đề quan trọng nhất để xây dựng phác đồ điều trị, lựa chọn dược mỹ phẩm tương thích, phòng ngừa các bệnh lý viêm da hoặc mụn trứng cá. Việc tự động hóa quy trình này bằng các kỹ thuật xử lý ảnh số mang lại tiềm năng ứng dụng to lớn trong các ứng dụng chẩn đoán trực tuyến và hệ thống gợi ý mỹ phẩm thông minh.

### 3.1.2. Cấu trúc tổ chức thư mục

Bộ dữ liệu trên đĩa lưu trữ được tổ chức theo chuẩn cấu trúc phân cấp thư mục hình ảnh phổ biến trong học máy (thường gọi là cấu trúc *ImageFolder* hay *Directory-based Partitioning*). Toàn bộ dữ liệu được phân chia sẵn thành ba tập con độc lập: tập huấn luyện (**train**), tập kiểm định (**valid** hay *validation*), và tập kiểm thử độc lập (**test**). Trong mỗi tập con, các tệp ảnh được phân loại vào các thư mục tương ứng với các nhãn lớp đích.

Cấu trúc cây thư mục chi tiết được biểu diễn như sau:

```text
skin_type_classification_dataset/ (hoặc data/)
├── train/
│   ├── combination/ (248 ảnh khuôn mặt da hỗn hợp dùng để huấn luyện)
│   ├── dry/         (833 ảnh khuôn mặt da khô dùng để huấn luyện)
│   ├── normal/      (976 ảnh khuôn mặt da thường dùng để huấn luyện)
│   └── oily/        (815 ảnh khuôn mặt da dầu dùng để huấn luyện)
├── valid/
│   ├── combination/ (63 ảnh khuôn mặt da hỗn hợp dùng để kiểm định)
│   ├── dry/         (257 ảnh khuôn mặt da khô dùng để kiểm định)
│   ├── normal/      (263 ảnh khuôn mặt da thường dùng để kiểm định)
│   └── oily/        (229 ảnh khuôn mặt da dầu dùng để kiểm định)
└── test/
    ├── combination/ (30 ảnh khuôn mặt da hỗn hợp dùng để kiểm thử độc lập)
    ├── dry/         (113 ảnh khuôn mặt da khô dùng để kiểm thử độc lập)
    ├── normal/      (160 ảnh khuôn mặt da thường dùng để kiểm thử độc lập)
    └── oily/        (106 ảnh khuôn mặt da dầu dùng để kiểm thử độc lập)
```

Quy cách tổ chức này mang lại tính chuẩn hóa cao, tương thích trực tiếp với các hàm nạp dữ liệu trong hệ thống mã nguồn của đề tài (`src.data_loader`) cũng như các thư viện học máy và thị giác máy tính phổ biến.

### 3.1.3. Định dạng ảnh và cơ chế gán nhãn

- **Định dạng tệp tin**: Toàn bộ **4,093** tệp ảnh trong bộ dữ liệu được lưu trữ dưới định dạng chuẩn **JPEG/JPG** (`.jpg`). Đây là định dạng lưu trữ ảnh kỹ thuật số tiêu chuẩn, sử dụng thuật toán nén có tổn thất (lossy compression) dựa trên biến đổi Cosine rời rạc (DCT), giúp tối ưu hóa dung lượng lưu trữ trong khi vẫn bảo toàn được các thông tin thị giác then chốt.
- **Không gian màu và độ sâu bit**: Các tệp ảnh được mã hóa theo không gian màu chuẩn **sRGB**, gồm 3 kênh màu độc lập đại diện cho các thành phần ánh sáng Đỏ (Red), Xanh lá (Green), và Xanh dương (Blue). Mỗi kênh màu sử dụng độ sâu 8-bit, tương ứng với dải giá trị cường độ nguyên rời rạc nằm trong khoảng $[0, 255]$, tạo nên độ sâu màu tổng cộng là 24-bit per pixel (khoảng 16.7 triệu màu sắc khả dĩ).
- **Kích thước ảnh**: Toàn bộ các ảnh trong bộ dữ liệu đã được chuẩn hóa kích thước không gian đồng nhất là **$640 \times 640$ pixel** với tỷ lệ khung hình vuông hoàn hảo ($1.0$).
- **Cơ chế gán nhãn (Labeling Mechanism)**: Bộ dữ liệu áp dụng cơ chế gán nhãn suy diễn theo đường dẫn thư mục mẹ (Directory-derived labeling). Nhãn của mỗi mẫu dữ liệu chính là tên thư mục con cấp thấp nhất chứa ảnh đó. Trong quy trình tiền xử lý của đề tài, các chuỗi ký tự nhãn này được ánh xạ sang dạng số nguyên định danh (Label Encoding) phục vụ cho các thuật toán học máy:
  - Nhãn `dry` $\rightarrow$ Giá trị mã hóa số học: `0`
  - Nhãn `normal` $\rightarrow$ Giá trị mã hóa số học: `1`
  - Nhãn `oily` $\rightarrow$ Giá trị mã hóa số học: `2`
  - *(Trường hợp mở rộng lớp `combination` được đánh chỉ số `3`)*.

---

## 3.2. KHẢO SÁT VÀ THỐNG KÊ CHI TIẾT BỘ DỮ LIỆU

### 3.2.1. Thống kê quy mô và thông số kỹ thuật ảnh

Kết quả chạy quy trình Giai đoạn 1 (`phase1_data_preparation.py`) trên toàn bộ tệp dữ liệu thực tế cho thấy các chỉ số kỹ thuật chi tiết của bộ dữ liệu:

*Bảng 3.1. Thống kê các thông số kỹ thuật thực nghiệm của bộ dữ liệu ảnh*

| Thông số kỹ thuật | Giá trị thống kê thực tế | Ghi chú kỹ thuật |
| :--- | :--- | :--- |
| **Tổng số lượng tệp ảnh** | **4,093** ảnh | Đã thẩm định $100\%$ hợp lệ, 0 ảnh lỗi |
| **Số lượng nhãn phân loại** | 04 lớp (`dry`, `normal`, `oily`, `combination`) | 3 lớp chính chiếm $91.67\%$ dữ liệu |
| **Định dạng lưu trữ** | 100% JPEG (`.jpg`) | Đọc chuẩn xác qua PIL và OpenCV |
| **Số kênh màu** | 3 kênh màu (sRGB) | 24-bit color depth (8-bit/kênh) |
| **Độ phân giải không gian** | Cố định **$640 \times 640$ pixel** | $W_{\min} = W_{\max} = 640$, $H_{\min} = H_{\max} = 640$ |
| **Tỷ lệ khung hình (Aspect Ratio)** | **$1.00$** | Ảnh vuông chuẩn hóa ($W / H = 1.0$) |
| **Dung lượng tệp nhỏ nhất** | **$9.79$ KB** | Mẫu nén dung lượng thấp |
| **Dung lượng tệp lớn nhất** | **$86.98$ KB** | Mẫu có độ chi tiết cao |
| **Dung lượng tệp trung bình** | **$40.94$ KB** | Trung vị: $38.93$ KB |
| **Tổng dung lượng đĩa** | **$163.63$ MB** | Thuận tiện cho xử lý lô và lưu cache |

Tính đồng nhất tuyệt đối về độ phân giải ($640 \times 640$ pixel) và tỷ lệ khung hình ($1:1$) của bộ dữ liệu mang lại ưu thế kỹ thuật rất lớn: loại bỏ hoàn toàn hiện tượng biến dạng méo hình học (aspect ratio distortion) khi thực hiện nội suy đổi kích thước (resizing) về kích thước chuẩn của pipeline ($224 \times 224$ pixel).

### 3.2.2. Số lượng ảnh từng lớp và phân tập dữ liệu

Số liệu thống kê chính xác được tổng hợp trên toàn bộ 4,093 hình ảnh và phân rã chi tiết qua ba tập con (Train, Valid, Test):

*Bảng 3.2. Thống kê số lượng ảnh chi tiết theo từng lớp và từng phân vùng thực tế*

| Lớp dữ liệu (Class) | Tên tiếng Việt | Tập Train | Tập Valid | Tập Test | Tổng số ảnh | Tỷ lệ phần trăm (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Da thường (Normal)** | Da thường | 976 | 263 | 160 | **1,399** | **34.18%** |
| **Da khô (Dry)** | Da khô | 833 | 257 | 113 | **1,203** | **29.39%** |
| **Da dầu (Oily)** | Da dầu / nhờn | 815 | 229 | 106 | **1,150** | **28.10%** |
| **Da hỗn hợp (Combination)** | Da hỗn hợp | 248 | 63 | 30 | **341** | **8.33%** |
| **Tổng cộng (Total)** | **Toàn bộ dataset** | **2,872** | **812** | **409** | **4,093** | **100.00%** |
| **Tỷ lệ phân vùng (%)** | - | **70.17%** | **19.84%** | **9.99%** | **100.00%** | - |

Nhận xét quan trọng về cơ cấu dữ liệu:
1. **Ba lớp sinh lý cốt lõi (`normal`, `dry`, `oily`)**: Chiếm tổng cộng **3,752 ảnh** (tương đương **$91.67\%$** toàn bộ dữ liệu). Quy mô mẫu giữa ba lớp này đạt mức độ cân bằng lý tưởng: lớp thường chiếm $34.18\%$, lớp khô chiếm $29.39\%$, và lớp dầu chiếm $28.10\%$.
2. **Lớp da hỗn hợp (`combination`)**: Chiếm **341 ảnh** (tương đương **$8.33\%$**). Đây là nhóm mẫu bổ trợ được bổ sung trong các phiên bản mở rộng của tập dữ liệu trên Kaggle. Do tính chất phức tạp về mặt sinh học (vừa có vùng dầu ở trán/mũi, vừa có vùng khô ở má), lớp này được phân tích độc lập nhằm phục vụ các mô hình phân loại đa nhãn nâng cao.

### 3.2.3. Đặc điểm thị giác và cơ chế phản xạ quang học của từng loại da

Dưới góc độ xử lý ảnh số và vật lý quang học, bề mặt da người là một cấu trúc tán xạ và phản xạ phức tạp bao gồm nhiều lớp sinh học (biểu bì, chân bì, lớp mỡ dưới da). Khi ánh sáng chiếu tới bề mặt da, hiện tượng tương tác quang học tuân theo mô hình phản xạ lai: kết hợp giữa **phản xạ khuếch tán Lambertian (Diffuse Reflection)** bắt nguồn từ sự tán xạ ánh sáng dưới bề mặt mô da, và **phản xạ gương Fresnel (Specular Reflection)** xảy ra tại bề mặt tiếp xúc trực tiếp giữa lớp màng lipid da và không khí. Bốn loại da biểu hiện những đặc trưng quang học và cấu trúc vi mô hoàn toàn khác biệt:

#### a) Lớp da khô (Dry Skin - 1,203 ảnh)
- **Đặc điểm mô học và hình thái học**: Da khô đặc trưng bởi sự suy giảm lipid nội bào và hàm lượng nước của lớp sừng (stratum corneum) dưới mức 10%. Các tế bào sừng chết kết tụ không đều và dễ bong tách, dẫn đến hiện tượng tróc vảy vi mô (micro-desquamation).
- **Đặc trưng thị giác trên ảnh số**:
  - *Độ bóng bề mặt (Glossiness)*: Gần như triệt tiêu hoàn toàn hiện tượng phản xạ gương. Ánh sáng chiếu vào bề mặt da khô bị tán xạ không định hướng theo quy luật phản xạ khuếch tán Lambertian, tạo cảm giác bề mặt lỳ (matte), xỉn màu và thiếu sức sống.
  - *Kết cấu bề mặt (Texture)*: Bề mặt thô ráp, xuất hiện các nếp nhăn nhỏ li ti dạng mạng lưới (fine dehydration lines), đặc biệt rõ ở vùng khóe mắt, khóe miệng và má.
  - *Lỗ chân lông (Pores)*: Lỗ chân lông rất nhỏ, khép kín, gần như không thể quan sát được trên ảnh ở độ phân giải $640 \times 640$.
  - *Màu sắc (Color tone)*: Thường có xu hướng nhợt nhạt hoặc xuất hiện các vùng đỏ cục bộ (erythema) do lớp màng lipid bảo vệ bị suy yếu làm lộ các vi mạch máu.

#### b) Lớp da thường (Normal Skin - 1,399 ảnh)
- **Đặc điểm mô học và hình thái học**: Đây là trạng thái cân bằng sinh lý lý tưởng (homeostasis) giữa lượng nước và dầu. Hàng rào bảo vệ da toàn vẹn, quá trình sừng hóa diễn ra bình thường theo chu kỳ sinh học.
- **Đặc trưng thị giác trên ảnh số**:
  - *Độ bóng bề mặt*: Ánh sáng phản xạ mềm mại, độ chói tự nhiên và phân bố đồng đều trên các đường nét giải phẫu của khuôn mặt mà không tạo ra các đốm chói lóa quá mức.
  - *Kết cấu bề mặt*: Cấu trúc vi mô của da rất mịn màng, vân da đồng nhất với tần số không gian biến thiên ở mức thấp đến trung bình, ít có các gồ ghề đột ngột.
  - *Lỗ chân lông*: Kích thước nhỏ mịn, phân bố đều đặn tại các vùng gò má và cánh mũi, không bị bít tắc hoặc phình đại.
  - *Màu sắc*: Sắc da tươi tắn, đồng đều giữa các vùng giải phẫu (trán, má, cằm), không có mảng đốm bất thường.

#### c) Lớp da dầu (Oily Skin - 1,150 ảnh)
- **Đặc điểm mô học và hình thái học**: Tuyến bã nhờn hoạt động quá mức (sebaceous gland hyperactivity), tiết ra lượng lớn chất bã nhờn (sebum) giàu triglyceride, squalene và wax ester tạo thành một màng lipid lỏng dày bao phủ bề mặt da.
- **Đặc trưng thị giác trên ảnh số**:
  - *Độ bóng bề mặt*: Xuất hiện các **điểm phản xạ gương chói lóa rõ rệt (Specular Highlights / Shininess)**. Tại các điểm này, cường độ pixel trên kênh độ sáng (Luminance $L$ trong LAB hoặc Value $V$ trong HSV) đạt tới giá trị bão hòa cực đại ($230 - 255$). Sự tương phản gay gắt giữa điểm chói sáng và vùng da lân cận là dấu hiệu thị giác nổi bật nhất của da dầu, tập trung chủ yếu ở vùng chữ T (trán, sống mũi, cằm) và đỉnh gò má.
  - *Kết cấu bề mặt*: Bề mặt da có độ thô ráp cao do sự giãn nở của các nang lông, tạo nên kết cấu gồ ghề dạng "vỏ quả cam" (orange-peel appearance).
  - *Lỗ chân lông*: Giãn to rõ rệt (enlarged / prominent pores), dễ thấy dưới dạng các lỗ sẫm màu li ti phân bố dày đặc ở vùng cánh mũi và hai bên má trong.
  - *Khuyết điểm đi kèm*: Thường xuất hiện mụn bọc, mụn đầu đen hoặc các vết thâm do bít tắc bã nhờn.

#### d) Lớp da hỗn hợp (Combination Skin - 341 ảnh)
- **Đặc điểm mô học và hình thái học**: Sự phân bố không đồng nhất của mật độ tuyến bã nhờn trên các vùng giải phẫu của khuôn mặt.
- **Đặc trưng thị giác trên ảnh số**: Vùng chữ T (trán, mũi, cằm) biểu hiện rõ nét các đặc tính của da dầu (bóng gương, lỗ chân lông to), trong khi hai bên vùng chữ U (gò má, quai hàm) lại mang đặc tính của da thường hoặc da khô (mịn lỳ hoặc hơi bong tróc). Do đó, việc trích xuất đặc trưng phân vùng (Region-Specific Features) giữa T-zone và U-zone đóng vai trò quyết định trong việc nhận diện loại da này.

### 3.2.4. Khảo sát chất lượng ảnh số và nhận diện các yếu tố gây nhiễu

Quá trình quét và thẩm định tự động trên tập dữ liệu đã phân tích sâu các chỉ số chất lượng ảnh số gồm: độ sắc nét Laplacian (Laplacian Variance), độ sáng trung bình (Mean Intensity), và tỷ lệ chói sáng:

- **Ảnh bị mờ nét (Blurry Images - Laplacian Variance $< 100.0$)**: Phát hiện **26 ảnh** ($0.64\%$). Các ảnh này bị suy giảm tần số không gian cao do rung máy hoặc mất nét, làm mờ các chi tiết kết cấu vi mô của da.
- **Ảnh thiếu sáng nghiêm trọng (Too Dark - Brightness $< 50.0$)**: Phát hiện **17 ảnh** ($0.42\%$). Cường độ điểm ảnh quá thấp khiến các đặc trưng màu sắc bị co hẹp về dải tối.
- **Ảnh cháy sáng / lóa sáng quá mức (Too Bright - Brightness $> 205.0$)**: Phát hiện **05 ảnh** ($0.12\%$). Vùng bão hòa trắng chiếm diện tích lớn có thể gây dương tính giả cho đặc trưng bóng dầu.
- **Tổng số ảnh cảnh báo chất lượng**: **46 ảnh** trên tổng số 4,093 ảnh (chỉ chiếm **$1.12\%$**).

Tỷ lệ ảnh cảnh báo chất lượng rất thấp ($1.12\%$) chứng minh bộ dữ liệu có chất lượng kỹ thuật cao, đáp ứng đầy đủ tiêu chuẩn cho việc xây dựng đường ống trích xuất đặc trưng và phân loại tự động.

---

## 3.3. PHÂN TÍCH PHÂN BỐ NHÃN VÀ MỨC ĐỘ CÂN BẰNG

### 3.3.1. Thống kê chi tiết tỷ lệ các lớp

Phân bố tần số mẫu giữa các lớp trên toàn bộ 4,093 hình ảnh được thể hiện định lượng:
- **Lớp Da thường (Normal)**: $1,399$ ảnh, chiếm **$34.18\%$**
- **Lớp Da khô (Dry)**: $1,203$ ảnh, chiếm **$29.39\%$**
- **Lớp Da dầu (Oily)**: $1,150$ ảnh, chiếm **$28.10\%$**
- **Lớp Da hỗn hợp (Combination)**: $341$ ảnh, chiếm **$8.33\%$**

```text
Biểu đồ phân bố số lượng mẫu giữa các lớp:
Normal      [========================================] 1,399 ảnh (34.18%)
Dry         [==================================]       1,203 ảnh (29.39%)
Oily        [=================================]        1,150 ảnh (28.10%)
Combination [==========]                                 341 ảnh  (8.33%)
```

### 3.3.2. Xác định và đo lường mức độ mất cân bằng dữ liệu

Trong lý thuyết học máy thống kê, mức độ mất cân bằng dữ liệu được định lượng thông qua **Hệ số mất cân bằng (Imbalance Ratio - $IR$)**, định nghĩa là tỷ số giữa số lượng mẫu của lớp đa số lớn nhất so với số lượng mẫu của lớp thiểu số:

1. **Khi xét trên 3 lớp chính (`normal`, `dry`, `oily` - quy mô 3,752 ảnh)**:
   $$IR_{\text{3-class}} = \frac{N_{\text{normal}}}{N_{\text{oily}}} = \frac{1399}{1150} \approx 1.2165$$
   $$IR_{\text{normal/dry}} = \frac{N_{\text{normal}}}{N_{\text{dry}}} = \frac{1399}{1203} \approx 1.1629$$

   **Nhận định**: Hệ số $IR \approx 1.22$ chứng minh ba lớp da chính đạt trạng thái **rất cân bằng (Well-Balanced)**. Trong học máy, hệ số $IR < 1.5$ được xem là dải phân bố lý tưởng, không làm méo mó hàm mục tiêu tối ưu hóa và không đòi hỏi các biện pháp can thiệp mất cân bằng cực đoan như SMOTE hay Random Undersampling.

2. **Khi xét toàn diện cả lớp `combination` (quy mô 4,093 ảnh)**:
   $$IR_{\text{all}} = \frac{N_{\text{normal}}}{N_{\text{combination}}} = \frac{1399}{341} \approx 4.1026$$
   Hệ số $IR \approx 4.10$ phản ánh mức mất cân bằng vừa phải (Moderate Imbalance) đối với riêng lớp `combination`.

### 3.3.3. Ảnh hưởng đến mô hình phân loại và giải pháp đề xuất

- **Đối với bài toán 3 lớp chính (Trọng tâm đề tài)**: Nhờ sự cân bằng tự nhiên ($34.18\% - 29.39\% - 28.10\%$), các mô hình phân loại SVM và KNN có thể hội tụ ổn định mà không lo ngại hiện tượng siêu phẳng bị dịch chuyển thiên lệch (decision boundary shift) hay hiện tượng lấn át láng giềng.
- **Trọng số lớp tự động (Balanced Class Weights)**: Khi huấn luyện SVM, trọng số lớp nghịch đảo tần suất $w_c = \frac{N}{C \cdot N_c}$ vẫn được áp dụng nhằm triệt tiêu hoàn toàn độ chênh lệch nhỏ còn lại giữa các lớp:
  - $w_{\text{normal}} = \frac{2872}{3 \times 976} \approx 0.9809$
  - $w_{\text{dry}} = \frac{2872}{3 \times 833} \approx 1.1492$
  - $w_{\text{oily}} = \frac{2872}{3 \times 815} \approx 1.1746$
- **Tiêu chuẩn đánh giá bắt buộc**: Áp dụng **Macro-averaged F1-Score** và **Balanced Accuracy** song hành cùng Accuracy tổng thể để đảm bảo đánh giá công bằng, khách quan trên từng lớp da.

---

## 3.4. QUY TRÌNH LÀM SẠCH, KIỂM ĐỊNH VÀ PHÁT HIỆN TRÙNG LẶP

### 3.4.1. Quy trình kiểm tra tính toàn vẹn và thẩm định tệp ảnh

Quy trình thẩm định toàn vẹn dữ liệu được thực thi tự động qua hàm `load_and_validate_dataset` trong module `src.data_loader`:
1. **Kiểm tra luồng tệp tin (PIL Verify)**: Mở và thẩm định cấu trúc header của từng ảnh bằng `PIL.Image.open().verify()`.
2. **Kiểm tra giải mã mảng pixel (OpenCV Decode)**: Giải mã mảng điểm ảnh thô bằng `cv2.imdecode` kết hợp đọc mảng byte nhị phân Unicode nhằm tương thích tuyệt đối với hệ điều hành Windows.
3. **Kiểm tra số chiều và ngưỡng kích thước**: Xác thực mảng có đúng 3 chiều (chiều cao, chiều rộng, 3 kênh màu) và kích thước không nhỏ hơn ngưỡng tối thiểu.

**Kết quả thẩm định thực tế**:
- Tổng số ảnh quét được: **4,093 tệp**.
- Số ảnh đọc thành công và hợp lệ: **4,093 / 4,093 tệp** ($100.00\%$).
- Số ảnh bị hỏng hóc hoặc lỗi giải mã: **0 tệp** ($0.00\%$).

### 3.4.2. Phát hiện ảnh trùng lặp bằng MD5 và Perceptual Hashing (dHash)

Trong các bộ dữ liệu thu thập từ Internet, hiện tượng trùng lặp ảnh (Duplicate Images) gây rủi ro nghiêm trọng về việc rò rỉ dữ liệu (Data Leakage) giữa tập Train và Test. Quy trình Phase 1 đã triển khai đồng thời hai kỹ thuật băm dữ liệu:
1. **Băm nhị phân tuyệt đối (MD5 Checksum)**: Tính mã băm MD5 128-bit trên toàn bộ luồng byte của tệp để phát hiện các tệp giống nhau $100\%$.
2. **Băm cảm nhận thị giác (Difference Hash - dHash)**: Thu nhỏ ảnh về kích thước $9 \times 8$ pixel mức xám, so sánh độ sáng các pixel liền kề theo hàng ngang để tạo chuỗi băm 64-bit phản ánh cấu trúc gradient thị giác, cho phép phát hiện các ảnh gần trùng lặp (Near-Duplicates) do nén lại hoặc thay đổi kích thước.

**Kết quả phát hiện trùng lặp thực tế**:
- **Ảnh trùng lặp tuyệt đối (MD5)**: Phát hiện **14 ảnh** thuộc **7 nhóm trùng lặp**. Đáng chú ý, phân tích mã băm đã phát hiện các trường hợp trùng lặp nhãn mâu thuẫn (Label Noise) và trùng xuyên phân tập:
  - Nhóm `05bec9...`: Cùng một ảnh nhưng được gán nhãn `dry` trong tập Train và nhãn `normal` trong tập Train (ảnh mang tên `normal_new_...`).
  - Nhóm `9d8e97...`: Cùng một ảnh nhưng xuất hiện ở tập Valid nhãn `normal` và tập Test nhãn `oily`.
  - Nhóm `5fd159...`: Xuất hiện đồng thời ở tập Train và tập Valid cùng nhãn `normal`.
  - Các nhóm `14d0b8...`, `ada31b...`, `af1509...`, `dba542...`: Các bản sao trùng lặp giữa các ảnh có tiền tố `_new_` hoặc hậu tố `-Copy`.
- **Ảnh tương đồng thị giác (dHash)**: Phát hiện **134 ảnh** có mã băm cảm nhận thị giác giống nhau.
- **Tổng số ảnh có dấu hiệu trùng lặp**: **148 ảnh**.

Toàn bộ thông tin này đã được lập chỉ mục và lưu vết chi tiết trong tệp siêu dữ liệu `outputs/dataset_metadata.csv` và `outputs/phase1_dataset_summary.json` để phục vụ khâu lọc sạch dữ liệu trước khi huấn luyện chính thức.

### 3.4.3. Đánh giá tính phân tầng của phân chia Train, Validation và Test

Nhóm nghiên cứu tiến hành kiểm tra tính phân tầng (**Stratified Cross-Tabulation Verification**) trên các phân vùng có sẵn của bộ dữ liệu:

*Bảng 3.3. So sánh tỷ lệ phần trăm phân bố nhãn thực tế giữa các tập dữ liệu*

| Tập dữ liệu (Subset) | Da khô (`dry`) | Da thường (`normal`) | Da dầu (`oily`) | Da hỗn hợp (`combination`) | Tổng số mẫu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Toàn bộ dataset** | **29.39%** ($1203/4093$) | **34.18%** ($1399/4093$) | **28.10%** ($1150/4093$) | **8.33%** ($341/4093$) | **4,093** (100%) |
| **Tập Train** | **29.00%** ($833/2872$) | **33.98%** ($976/2872$) | **28.38%** ($815/2872$) | **8.64%** ($248/2872$) | **2,872** (70.2%) |
| **Tập Valid** | **31.65%** ($257/812$) | **32.39%** ($263/812$) | **28.20%** ($229/812$) | **7.76%** ($63/812$) | **812** (19.8%) |
| **Tập Test** | **27.63%** ($113/409$) | **39.12%** ($160/409$) | **25.92%** ($106/409$) | **7.33%** ($30/409$) | **409** (10.0%) |

**Đánh giá**:
- Tỷ lệ phân chia tổng thể đạt chuẩn xấp xỉ **$70\% : 20\% : 10\%$** cho Train, Valid và Test.
- Tỷ lệ phần trăm của từng lớp trên tập Train và Valid dao động rất sát với tỷ lệ phân bố tổng thể (sai lệch dưới $2.2\%$), bảo toàn tốt tính đại diện thống kê của từng nhóm da.

### 3.4.4. Các nguyên tắc ngăn ngừa rò rỉ dữ liệu (Data Leakage Prevention)

Đề tài áp dụng nghiêm ngặt 4 nguyên tắc bảo toàn dữ liệu:
1. **Phong tỏa tập kiểm thử (Strict Hold-Out Test Set)**: Toàn bộ 409 ảnh trong tập `test` được cô lập hoàn toàn, không tham gia vào bất kỳ khâu nào của quá trình chọn mô hình, tinh chỉnh siêu tham số hay huấn luyện.
2. **Chuẩn hóa đặc trưng không rò rỉ (Leakage-Free Feature Normalization)**: Bộ chuẩn hóa `FeatureNormalizer` (`StandardScaler` và `PCA`) chỉ được `fit` trên ma trận đặc trưng của tập Train (`normalizer.fit(X_train)`), sau đó áp dụng phép `transform` trên Valid và Test.
3. **Tăng cường dữ liệu chỉ trên tập Train (Train-Only Augmentation)**: Mọi thao tác tăng cường mẫu (nếu áp dụng) chỉ thực hiện trên tập Train.
4. **Khử trùng lặp xuyên tập (Cross-Partition Deduplication)**: Dựa trên danh sách các nhóm trùng MD5 đã phát hiện ở mục 3.4.2, hệ thống loại bỏ các ảnh trùng lặp giữa tập Train và tập Test để đảm bảo tính độc lập thống kê.

---

## 3.5. ĐỊNH HƯỚNG THỰC NGHIỆM VÀ KẾT QUẢ ĐƯỜNG ỐNG XỬ LÝ (PHASE 2 PIPELINE)

Dựa trên đặc tính dữ liệu đã khảo sát, Giai đoạn 2 (`phase2_build_pipeline.py`) đã xây dựng và kiểm thử thành công hai đường ống xử lý ảnh hoàn chỉnh: **BaselinePipeline** và **EnhancedPipeline**.

### 3.5.1. Kiến trúc hai đường ống xử lý thực nghiệm

```text
Sơ đồ khối hai đường ống xử lý:

[BASELINE PIPELINE]:
Ảnh thô (640x640) ──> Resize (224x224) ──> Phân vùng da HSV thô ──> Trích xuất 126 chiều ──> Đặc trưng Baseline

[ENHANCED PIPELINE]:
Ảnh thô (640x640) ──> Resize (224x224)
                  ──> Khử nhiễu Gaussian (5x5, sigma=1.0)
                  ──> Tăng tương phản thích ứng CLAHE (kênh L, clip=2.0)
                  ──> Phân vùng da đa không gian (HSV AND YCrCb)
                  ──> Lọc hình thái học (Morphology Closing + Opening + Hole Filling)
                  ──> Tách vùng da sạch (Skin ROI)
                  ──> Trích xuất 126 đặc trưng (Màu sắc, Độ sáng, LBP)
                  ──> Chuẩn hóa Z-Score & Giảm chiều PCA (13 chiều, 95.16% var)
```

1. **Đường ống cơ sở (Baseline Pipeline)**:
   - Đọc ảnh $\rightarrow$ Resize trực tiếp về $224 \times 224$ pixel $\rightarrow$ Chuyển không gian màu sang HSV và áp dụng ngưỡng thô ($H \in [0, 50]$, $S \in [0.23, 0.68]$, $V \ge 0.35$) $\rightarrow$ Trích xuất đặc trưng.
2. **Đường ống nâng cao (Enhanced Pipeline)**:
   - Đọc ảnh $\rightarrow$ Resize về $224 \times 224$ pixel.
   - **Khử nhiễu**: Bộ lọc Gaussian kernel kích thước $(5, 5)$, độ lệch chuẩn $\sigma = 1.0$ giúp làm mịn nhiễu hạt cảm biến nhưng giữ lại viền giải phẫu.
   - **Cân bằng sáng thích ứng CLAHE**: Áp dụng thuật toán CLAHE với ngưỡng cắt clip limit $2.0$ và lưới ô $8 \times 8$ trên kênh độ sáng (Luminance $L$ trong LAB), triệt tiêu hiệu ứng chiếu sáng một phía.
   - **Phân đoạn da đa không gian màu**: Kết hợp phép giao logic (AND) giữa ngưỡng màu trong không gian HSV và không gian YCrCb ($Y \ge 80$, $Cr \in [133, 173]$, $Cb \in [77, 127]$).
   - **Lọc sạch mặt nạ hình thái học**: Áp dụng phép đóng (Closing) với phần tử cấu trúc elip $5 \times 5$ để nối liền các mảng da đứt gãy, tiếp theo là phép mở (Opening) để loại bỏ nhiễu cô lập, và thuật toán lũy tích đường viền (flood fill) để lấp đầy các lỗ thủng bên trong vùng da mặt.
   - **Trích xuất vùng ROI da**: Áp dụng mặt nạ đã làm sạch lên ảnh gốc để tách riêng mảng da mặt tinh khiết, loại bỏ hoàn toàn mắt, tóc và hậu cảnh.

### 3.5.2. Cấu trúc không gian đặc trưng trích xuất (126 chiều)

Vector đặc trưng hoàn chỉnh $X \in \mathbb{R}^{126}$ bao gồm 3 nhóm thành phần được thiết kế chuyên biệt cho bài toán phân loại da:
- **Nhóm A: Đặc trưng màu sắc (Color Features - 105 chiều)**:
  - *Mômen màu thống kê (Color Moments - 27 chiều)*: Giá trị trung bình ($\mu$), độ lệch chuẩn ($\sigma$), và hệ số bất đối xứng ($s$) tính trên từng kênh của 3 không gian màu RGB, HSV và YCrCb ($3 \times 3 \times 3 = 27$ chiều).
  - *Lược đồ màu (Color Histograms - 72 chiều)*: Lược đồ 8-bin trên từng kênh của 3 không gian màu ($3 \times 3 \times 8 = 72$ chiều), nắm bắt phân bố tần suất sắc độ.
  - *Màu chủ đạo (Dominant Colors - 6 chiều)*: Tọa độ RGB của 2 cụm màu trung tâm xác định qua thuật toán K-Means ($2 \times 3 = 6$ chiều).
- **Nhóm B: Đặc trưng độ sáng và độ bóng nhờn (Brightness & Highlights - 7 chiều)**:
  - Cường độ sáng trung bình, trung vị, giá trị nhỏ nhất, giá trị lớn nhất, độ tương phản RMS trên kênh xám.
  - **Tỷ lệ vùng chói bóng nhờn (Specular Highlight Ratio)**: Tỷ lệ diện tích pixel có $V > 230$ và $S < 35$ so với tổng diện tích da, là đặc trưng chỉ điểm then chốt của da dầu.
  - Độ chênh lệch độ sáng cực đại giữa vùng chữ T và hai bên má.
- **Nhóm C: Đặc trưng kết cấu vi mô LBP (Texture LBP - 14 chiều)**:
  - Lược đồ LBP đồng nhất (Uniform LBP $\text{LBP}_{8, 1}^{u2}$ với bán kính $R=1$, số điểm láng giềng $P=8$) gồm 10 bins biểu diễn tần suất các mẫu vi kết cấu.
  - 4 chỉ số thống kê kết cấu LBP: Giá trị trung bình, độ lệch chuẩn, năng lượng (Energy) và Entropi kết cấu.

### 3.5.3. Kết quả đánh giá so sánh định lượng thực nghiệm (Baseline vs Enhanced)

Kết quả thực nghiệm đo đạc chi tiết trên tập mẫu thử nghiệm thông qua kịch bản `phase2_build_pipeline.py` được tổng hợp trong Bảng 3.4:

*Bảng 3.4. So sánh định lượng hiệu năng giữa Baseline Pipeline và Enhanced Pipeline*

| Tiêu chí so sánh thực nghiệm | Baseline Pipeline | Enhanced Pipeline | Đánh giá cải thiện / Nhận xét |
| :--- | :---: | :---: | :--- |
| **Thời gian thực thi trung bình/ảnh** | **$60.77$ ms** | **$61.63$ ms** | Độ trễ tăng thêm chỉ $\sim 0.86$ ms ($+1.4\%$), hoàn toàn đáp ứng thời gian thực (Real-time $> 16$ FPS) |
| **Tỷ lệ diện tích da trích xuất trung bình** | **$32.53\%$** | **$36.24\%$** | **Tăng $+3.71\%$ diện tích da hữu ích** nhờ cơ chế phối hợp đa không gian và lấp lỗ hổng |
| **Tỷ lệ mặt nạ da đạt chuẩn ($>5\%$ diện tích)** | **$93.33\%$** | **$93.33\%$** | Độ ổn định cao, lọc bỏ chính xác các ảnh không chứa da |
| **Khử nhiễu & Làm mịn hạt cảm biến** | Không hỗ trợ | Gaussian Kernel ($5 \times 5, \sigma=1.0$) | Triệt tiêu nhiễu bề mặt, bảo toàn biên vân da |
| **Cân bằng tương phản thích ứng** | Không hỗ trợ | CLAHE (Clip $2.0$, Grid $8 \times 8$) | Khắc phục hoàn toàn hiện tượng lệch sáng một phía |
| **Loại bỏ vùng ngoại lai (tóc, nền, mắt)** | Kém (Mặt nạ thô) | Xuất sắc (Morphology + Mask) | Tách riêng mảng da mặt tinh khiết phục vụ trích xuất |
| **Số chiều vector đặc trưng thô** | 126 chiều | 126 chiều | Đầy đủ 3 nhóm A (Màu), B (Bóng nhờn), C (LBP) |
| **Giảm chiều đặc trưng qua PCA** | - | **13 chiều** | **Giữ lại $95.16\%$ tổng phương sai**, giảm $89.7\%$ số chiều |

### 3.5.4. Các biểu đồ trực quan hóa đã xuất bản trong thực nghiệm

Toàn bộ quy trình thực nghiệm đã tự động xuất bản các biểu đồ phân tích chất lượng cao vào thư mục `outputs/figures/`:
1. `class_distribution.png`: Biểu đồ cột và biểu đồ tròn thể hiện phân bố mẫu giữa các lớp và phân tập dữ liệu.
2. `sample_images_grid.png`: Lưới ảnh chân dung khuôn mặt thực tế đại diện cho từng loại da.
3. `image_size_distribution.png`: Biểu đồ tán xạ phân bố độ phân giải ($640 \times 640$) và tỷ lệ khung hình ($1.0$).
4. `image_quality_distribution.png`: Phân bố độ sắc nét Laplacian, độ sáng trung bình và tỷ lệ bóng nhờn giữa các lớp da.
5. `phase2_step2_baseline_preprocessing.png`: Minh họa trực quan quá trình resize trực tiếp của Baseline Pipeline.
6. `phase2_step3_enhanced_preprocessing.png`: Minh họa hiệu quả của bộ lọc Gaussian và cân bằng sáng thích ứng CLAHE.
7. `phase2_step4_skin_detection_comparison.png`: So sánh mặt nạ phân đoạn da giữa HSV, YCrCb, phép giao AND và sau khi làm sạch hình thái học.
8. `phase2_step5_color_spaces.png`: Trực quan hóa ảnh qua 5 không gian màu (RGB, HSV, YCrCb, LAB, Grayscale).
9. `phase2_step6_features_breakdown.png`: Biểu đồ phân rã chi tiết 126 chiều đặc trưng của Nhóm A, Nhóm B và Nhóm C.
10. `phase2_step9_pipeline_comparison.png`: Biểu đồ so sánh định lượng tổng thể thời gian và chất lượng giữa hai đường ống.
11. `phase2_single_image_comparison.png`: Lưới so sánh trực diện $2 \times 4$ trên cùng một ảnh mẫu giữa Baseline và Enhanced Pipeline.

---

## TỔNG KẾT CHƯƠNG 3

Chương 3 đã hoàn thành toàn diện công tác khảo sát thực nghiệm, làm sạch và phân tích lý thuyết đối với bộ dữ liệu thực tế **"Oily, Dry and Normal Skin Types Dataset"** với các kết quả cụ thể:

1. **Xác lập chính xác quy mô và thông số dữ liệu**: Khảo sát toàn bộ **4,093 hình ảnh JPEG** chuẩn hóa $640 \times 640$ pixel, 24-bit sRGB với tổng dung lượng $163.63$ MB. Dữ liệu được tổ chức theo 3 tập: Train ($2,872$ ảnh - $70.2\%$), Valid ($812$ ảnh - $19.8\%$) và Test ($409$ ảnh - $10.0\%$).
2. **Khẳng định tính cân bằng lý tưởng của ba lớp da chính**: Ba lớp da trọng tâm nghiên cứu (`normal`: 1,399 ảnh, `dry`: 1,203 ảnh, `oily`: 1,150 ảnh) chiếm **$91.67\%$** toàn bộ dữ liệu với hệ số mất cân bằng $IR \approx 1.22$, tạo điều kiện cực kỳ thuận lợi cho việc học các đặc trưng phân loại. Lớp da hỗn hợp (`combination`: 341 ảnh - $8.33\%$) được khảo sát như một bài toán mở rộng phân loại đa nhãn.
3. **Thẩm định chất lượng và khử trùng lặp nghiêm ngặt**: Xác nhận $100\%$ ảnh hợp lệ về mặt giải mã tập tin. Phát hiện và khoanh vùng 14 ảnh trùng lặp tuyệt đối (MD5) và 134 ảnh trùng lặp cảm nhận (dHash) để ngăn ngừa rò rỉ dữ liệu. Tỷ lệ ảnh cảnh báo chất lượng mờ/tối/sáng chỉ chiếm $1.12\%$.
4. **Xây dựng và kiểm chứng thành công hai đường ống xử lý ảnh (Phase 2)**:
   - Chứng minh đường ống nâng cao (**Enhanced Pipeline**) vượt trội so với đường ống cơ sở (**Baseline Pipeline**): Tăng tỷ lệ bóc tách diện tích da hữu ích từ $32.53\%$ lên $36.24\%$ trong khi thời gian xử lý vẫn duy trì ở mức tối ưu ($61.63$ ms/ảnh, tương đương tốc độ xử lý thời gian thực).
   - Thiết kế thành công không gian **126 đặc trưng trích xuất** (105 đặc trưng màu sắc, 7 đặc trưng bóng sáng/T-zone, 14 đặc trưng kết cấu LBP) và nén tối ưu xuống **13 chiều** qua PCA mà vẫn bảo toàn **$95.16\%$** thông tin phương sai.

Toàn bộ các phân tích định lượng, bộ dữ liệu đã làm sạch và các ma trận đặc trưng trích xuất trong chương này là cơ sở trực tiếp để triển khai huấn luyện, tối ưu hóa siêu tham số cho các bộ phân loại KNN và SVM trong **Chương 4: Phương pháp đề xuất và cài đặt thực nghiệm**.
