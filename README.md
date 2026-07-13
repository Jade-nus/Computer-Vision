# DroneVisionPro - Hệ thống Xử lý Ảnh cho Drone

**DroneVisionPro** là hệ thống xử lý ảnh thời gian thực được thiết kế riêng cho ứng dụng drone. Dự án hỗ trợ cả 2 ngôn ngữ **Python** (dành cho việc chạy thử nghiệm siêu tốc) và **C++** (dành cho nhúng vào phần cứng thực tế với hiệu năng cao nhất).

Hệ thống tích hợp nhiều module chuyên dụng cho các tác vụ thị giác máy tính:
- **ArUco Detector:** Nhận diện bãi đáp hạ cánh chính xác đến từng sub-pixel.
- **QR Detector:** Quét mã QR siêu nhạy từ xa (Sử dụng PyZbar).
- **Object Detector:** Nhận diện 80 loại vật thể bằng AI (YOLOv4-tiny).
- **Drone HUD:** Giao diện bay chuyên nghiệp với Crosshair, FPS, Altitude.

---

## 📦 TÀI NGUYÊN BẮT BUỘC (Dành cho cả C++ và Python)

Để tính năng Trí tuệ nhân tạo (YOLO) hoạt động, bạn **bắt buộc** phải có 3 file mô hình sau trong thư mục `assets/models/` (Hiện tại đã được tải sẵn):
1. `yolov4-tiny.cfg` (File cấu trúc mạng nơ-ron)
2. `yolov4-tiny.weights` (File trọng số đã huấn luyện)
3. `coco.names` (Danh sách 80 loại vật thể)

---

## 🚀 CÁCH 1: CHẠY BẢN PYTHON (Siêu Nhanh - Khuyên dùng)

Đây là cách tuyệt vời nhất để test thuật toán trực tiếp trên Camera Laptop mà không cần biên dịch phức tạp.

### 1. Yêu cầu tải & Cài đặt
Bạn chỉ cần mở Terminal (PowerShell) của VS Code trong thư mục dự án và gõ lệnh sau để tải các thư viện cần thiết (Chỉ tốn ~50MB):

```powershell
pip install opencv-python opencv-contrib-python numpy pyzbar
```

### 2. Cách chạy
Mở VS Code, nhấn phím **F5** (hoặc Run -> Start Debugging) để chạy file cấu hình đã được thiết lập sẵn, hoặc gõ thủ công lệnh sau:

```powershell
cd python_app
python main.py
```

**Thao tác Camera:**
- Phím `q`: Thoát Camera an toàn.
- Hoặc bấm dấu `X` trên cửa sổ Camera.

---

## 🛠️ CÁCH 2: BUILD BẢN C++ (Chuyên sâu - Hiệu năng cao)

Bản C++ chứa mã nguồn khổng lồ (>1000 dòng code) được chia thành các module chuyên nghiệp. Để build được bản C++, bạn cần phải cài đặt trình biên dịch đầy đủ.

### 1. Yêu cầu tải & Cài đặt
Bạn cần tải và cài đặt một trong hai bộ công cụ sau (Dung lượng lớn):

**Tùy chọn A: Dành cho người dùng Visual Studio (Khuyên dùng trên Windows)**
- Tải [Visual Studio 2022 Community](https://visualstudio.microsoft.com/vs/community/)
- Khi cài, nhớ tích chọn **"Desktop development with C++"**.

**Tùy chọn B: Dành cho người dùng MinGW/CodeBlocks**
- Đảm bảo máy đã cài đặt `CMake` và `MinGW-w64` (GCC).
- Bạn PHẢI tự biên dịch (compile) thư viện OpenCV từ mã nguồn gốc để có được module `opencv_contrib` (chứa ArUco) tương thích với MinGW. 

### 2. Cách Build (Ví dụ với Visual Studio)

Mở Terminal tại thư mục gốc của dự án `DroneVisionPro`:

```powershell
# Tạo thư mục build
mkdir build
cd build

# Sinh file project Visual Studio bằng CMake
cmake ..

# Biên dịch chương trình (Sẽ tốn chút thời gian)
cmake --build . --config Release

# Chạy phần mềm
.\Release\DroneVisionPro.exe
```

---

## 📚 CẤU TRÚC MÃ NGUỒN CỦA DỰ ÁN

```text
DroneVisionPro/
├── python_app/
│   └── main.py                 # Mã nguồn phiên bản Python (Chạy ngay)
├── include/                    # Header files của C++
│   ├── aruco_detector.hpp      # Nhận diện bãi đáp (ArUco)
│   ├── qr_detector.hpp         # Quét mã QR
│   ├── object_detector.hpp     # Nhận diện vật thể AI (YOLO)
│   ├── image_processor.hpp     # Bộ lọc và xử lý ảnh
│   ├── drone_hud.hpp           # Giao diện bay giả lập
│   └── utils.hpp               
├── src/                        # Source files của C++
│   ├── main.cpp                # File chạy chính của C++
│   ├── aruco_detector.cpp      
│   ├── ...                     # (Các file .cpp tương ứng)
├── assets/
│   └── models/                 # Thư mục chứa mô hình YOLO (cfg, weights)
├── CMakeLists.txt              # Cấu hình build cho C++
└── README.md                   # File hướng dẫn này
```

---
*Dự án được phát triển và tối ưu hóa cho hệ thống điều khiển Drone thời gian thực!*
