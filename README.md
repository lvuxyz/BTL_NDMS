# Xây dựng hệ thống nhận diện màu sắc bằng Pandas và OpenCV
Dự án này triển khai một hệ thống nhận dạng màu sắc thời gian thực sử dụng thuật toán K-Nearest Neighbors (KNN) và histogram màu. Hệ thống có thể phát hiện và phân loại màu sắc trong cả ảnh tĩnh và luồng video thời gian thực thông qua webcam.

## Tính Năng

- Phát hiện màu sắc thời gian thực qua webcam
- Phân tích màu sắc trên ảnh tĩnh
- Hỗ trợ nhiều lớp màu (Đỏ, Xanh lá, Xanh dương, Vàng, Cam, Trắng, Đen, Tím)
- Giao diện người dùng thân thiện được xây dựng bằng Tkinter
- Phát hiện đối tượng sử dụng YOLOv3
- Phân loại màu sắc sử dụng thuật toán KNN
- Trích xuất đặc trưng histogram màu

## Yêu Cầu Hệ Thống

### Thư viện chính
1. **OpenCV (opencv-python)**
   - Sử dụng để xử lý ảnh và video
   - Trích xuất đặc trưng histogram màu
   - Phát hiện đối tượng với YOLOv3
   - Thao tác với webcam
   - Phiên bản khuyến nghị: >= 4.5.0

2. **Pandas**
   - Đọc và xử lý dữ liệu huấn luyện
   - Quản lý dữ liệu màu sắc
   - Xử lý tệp CSV chứa thông tin màu
   - Phiên bản khuyến nghị: >= 1.3.0

### Các thư viện phụ thuộc khác
```bash
numpy       # Xử lý mảng và tính toán
scikit-learn # Đánh giá mô hình và độ chính xác
pillow      # Xử lý ảnh
tkinter     # Giao diện đồ họa

```

Bạn cũng cần các file mô hình sau:
- yolov3.weights
- yolov3.cfg
- coco.names

## Cấu Trúc Dự Án

```
├── src/
│   ├── color_recognition_api/
│   │   ├── color_histogram_feature_extraction.py
│   │   └── knn_classifier.py
│   ├── click_image.py
│   ├── color_classification_webcam.py
│   └── test.py
├── training_dataset/
│   ├── red/
│   ├── blue/
│   ├── green/
│   ├── yellow/
│   ├── orange/
│   ├── white/
│   ├── black/
│   └── violet/
└── tkinter_main.py
```

## Cách Hoạt Động

1. **Trích Xuất Đặc Trưng Màu**
   - Sử dụng histogram màu để trích xuất đặc trưng RGB từ ảnh
   - Xử lý cả ảnh huấn luyện và ảnh kiểm thử
   - Tạo vector đặc trưng cho phân loại

2. **Phân Loại KNN**
   - Triển khai thuật toán K-Nearest Neighbors
   - Sử dụng khoảng cách Euclidean để đo độ tương đồng
   - Phân loại màu sắc dựa trên dữ liệu huấn luyện
   - Hỗ trợ điều chỉnh giá trị K (mặc định: k=7)

3. **Phát Hiện Đối Tượng**
   - Sử dụng YOLOv3 để phát hiện đối tượng thời gian thực
   - Nhận diện đối tượng trong luồng video
   - Tạo khung bao quanh đối tượng được phát hiện

4. **Giao Diện GUI**
   - Cung cấp giao diện dễ sử dụng cho:
     - Tải ảnh tĩnh
     - Truy cập luồng webcam
     - Hiển thị kết quả
   - Hiển thị dự đoán màu sắc và điểm tin cậy

## Cách Sử Dụng

1. **Chạy Ứng Dụng**
   ```bash
   python tkinter_main.py
   ```

2. **Sử Dụng Chế Độ Ảnh Tĩnh**
   - Nhấp vào nút "Nhập ảnh"
   - Chọn file ảnh
   - Nhấp đúp vào bất kỳ điểm nào trong ảnh để lấy thông tin màu sắc

3. **Sử Dụng Chế Độ Webcam**
   - Nhấp vào nút "Bật Camera"
   - Hướng camera vào đối tượng
   - Phát hiện màu sắc thời gian thực sẽ được hiển thị

4. **Thoát Ứng Dụng**
   - Nhấp vào nút "Thoát" hoặc nhấn ESC

## Huấn Luyện

Hệ thống đi kèm với dữ liệu đã được huấn luyện sẵn, nhưng bạn có thể thêm ảnh huấn luyện của riêng mình:

1. Thêm ảnh vào các thư mục màu tương ứng trong `training_dataset/`
2. Chạy huấn luyện:
   ```bash
   python test.py
   ```

## Đánh Giá Hiệu Suất

Hệ thống bao gồm các chỉ số đánh giá:
- Điểm F1
- Độ chính xác (Precision)
- Độ thu hồi (Recall)
- Ma trận nhầm lẫn (Confusion Matrix)

Bạn có thể chạy đánh giá bằng cách:
```bash
python test.py
```

## Hạn Chế

- Độ chính xác phát hiện màu phụ thuộc vào điều kiện ánh sáng
- Kết quả tốt nhất đạt được với màu đồng nhất
- Hiệu suất có thể thay đổi với các mẫu phức tạp
- Yêu cầu chất lượng webcam tốt để phát hiện thời gian thực chính xác

## Đóng Góp

Hãy tự do fork dự án này và gửi pull request. Đối với những thay đổi lớn, vui lòng mở issue trước để thảo luận về những gì bạn muốn thay đổi.

## Giấy Phép

Dự án này được cấp phép theo Giấy phép MIT - xem file LICENSE để biết chi tiết.
