// ==============================================================================
// DroneVisionPro - Hệ thống Xử lý Ảnh Chuyên nghiệp cho Drone
// ==============================================================================
// File: main.cpp
// Mô tả: File chính của chương trình, chứa menu điều khiển và các chế độ demo
// Tác giả: DroneVisionPro Team
// ==============================================================================

// --- Nạp các thư viện chuẩn C++ ---
#include <iostream>      // Thư viện nhập/xuất (cout, cin)
#include <string>        // Thư viện xử lý chuỗi
#include <chrono>        // Thư viện thời gian (đo FPS, timestamp)
#include <thread>        // Thư viện đa luồng (sleep)
#include <filesystem>    // Thư viện quản lý file/thư mục (C++17)

// --- Nạp thư viện OpenCV ---
#include <opencv2/opencv.hpp>    // Thư viện OpenCV chính (core, imgproc, highgui)

// --- Nạp các module của dự án ---
#include "aruco_detector.hpp"    // Module phát hiện ArUco marker
#include "qr_detector.hpp"       // Module phát hiện QR code
#include "object_detector.hpp"   // Module phát hiện đối tượng (YOLO)
#include "image_processor.hpp"   // Module xử lý ảnh (filters)
#include "drone_hud.hpp"         // Module HUD overlay cho drone
#include "utils.hpp"             // Module tiện ích chung

// --- Sử dụng namespace để viết code ngắn gọn hơn ---
using namespace std;             // Cho phép dùng cout, cin thay vì std::cout
using namespace cv;              // Cho phép dùng Mat, VideoCapture thay vì cv::Mat
namespace fs = std::filesystem;  // Rút gọn tên namespace filesystem

// ==============================================================================
// Hàm hiển thị menu chính của chương trình
// ==============================================================================
void showMainMenu() {
    // Xóa màn hình console (Windows)
    #ifdef _WIN32
        system("cls");          // Lệnh xóa màn hình trên Windows
    #else
        system("clear");        // Lệnh xóa màn hình trên Linux/Mac
    #endif

    // In banner và menu với khung viền đẹp
    cout << "\n";
    cout << "  ╔══════════════════════════════════════════════════════════╗\n";
    cout << "  ║                                                        ║\n";
    cout << "  ║          DRONE VISION PRO v1.0                         ║\n";
    cout << "  ║   He thong Xu ly Anh Chuyen nghiep cho Drone           ║\n";
    cout << "  ║                                                        ║\n";
    cout << "  ╠══════════════════════════════════════════════════════════╣\n";
    cout << "  ║                                                        ║\n";
    cout << "  ║   [1]  ArUco Marker Detection                         ║\n";
    cout << "  ║        Phat hien ArUco marker, uoc tinh vi tri 3D     ║\n";
    cout << "  ║                                                        ║\n";
    cout << "  ║   [2]  QR Code Scanner                                ║\n";
    cout << "  ║        Quet va giai ma QR code thoi gian thuc         ║\n";
    cout << "  ║                                                        ║\n";
    cout << "  ║   [3]  Object Detection (YOLO)                        ║\n";
    cout << "  ║        Phat hien doi tuong bang YOLO DNN               ║\n";
    cout << "  ║                                                        ║\n";
    cout << "  ║   [4]  Full Mission Mode                              ║\n";
    cout << "  ║        Ket hop tat ca module + HUD overlay            ║\n";
    cout << "  ║                                                        ║\n";
    cout << "  ║   [5]  Image Processing Lab                           ║\n";
    cout << "  ║        Thu nghiem cac bo loc xu ly anh                 ║\n";
    cout << "  ║                                                        ║\n";
    cout << "  ║   [6]  ArUco Marker Generator                         ║\n";
    cout << "  ║        Tao anh ArUco marker de in                     ║\n";
    cout << "  ║                                                        ║\n";
    cout << "  ║   [0]  Thoat chuong trinh                             ║\n";
    cout << "  ║                                                        ║\n";
    cout << "  ╚══════════════════════════════════════════════════════════╝\n";
    cout << "\n  >> Chon che do (0-6): ";
}

// ==============================================================================
// Hàm mở camera hoặc video file
// Trả về VideoCapture đã mở thành công
// ==============================================================================
VideoCapture openCamera() {
    VideoCapture cap;        // Tạo đối tượng VideoCapture để đọc video/camera

    // Hỏi người dùng chọn nguồn video
    cout << "\n  Chon nguon video:\n";
    cout << "  [1] Webcam (camera 0)\n";
    cout << "  [2] Video file\n";
    cout << "  [3] IP Camera / RTSP stream\n";
    cout << "  >> Lua chon: ";

    int source_choice;       // Biến lưu lựa chọn của người dùng
    cin >> source_choice;    // Đọc lựa chọn từ bàn phím
    cin.ignore();            // Xóa ký tự newline còn sót trong buffer

    switch (source_choice) {
        case 1: {
            // --- Mở webcam mặc định (index 0) ---
            drone_utils::logInfo("Dang mo webcam...");    // In thông báo
            cap.open(0, cv::CAP_DSHOW);  // Mở camera 0 với DirectShow backend (Windows)

            if (!cap.isOpened()) {
                // Nếu DirectShow thất bại, thử backend mặc định
                cap.open(0);    // Thử mở lại với backend tự động
            }
            break;
        }
        case 2: {
            // --- Mở video file ---
            cout << "  Nhap duong dan video file: ";
            string video_path;                // Biến lưu đường dẫn file
            getline(cin, video_path);         // Đọc đường dẫn (có thể chứa dấu cách)

            drone_utils::logInfo("Dang mo video: " + video_path);
            cap.open(video_path);             // Mở file video
            break;
        }
        case 3: {
            // --- Mở IP Camera hoặc RTSP stream ---
            cout << "  Nhap URL stream (vi du: rtsp://...): ";
            string stream_url;               // Biến lưu URL stream
            getline(cin, stream_url);        // Đọc URL

            drone_utils::logInfo("Dang ket noi stream: " + stream_url);
            cap.open(stream_url);            // Mở stream từ URL
            break;
        }
        default: {
            // --- Mặc định mở webcam ---
            drone_utils::logWarning("Lua chon khong hop le, su dung webcam mac dinh");
            cap.open(0);                     // Mở camera 0
            break;
        }
    }

    // Kiểm tra xem camera/video có mở thành công không
    if (!cap.isOpened()) {
        drone_utils::logError("Khong the mo nguon video!");   // Báo lỗi
        return cap;                                            // Trả về cap chưa mở
    }

    // Thiết lập độ phân giải mặc định cho camera
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);    // Đặt chiều rộng frame = 1280 pixels
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);    // Đặt chiều cao frame = 720 pixels

    // In thông tin camera
    int frame_w = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));   // Lấy chiều rộng thực tế
    int frame_h = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));  // Lấy chiều cao thực tế
    double fps = cap.get(CAP_PROP_FPS);                               // Lấy FPS của nguồn

    // Hiển thị thông tin nguồn video
    drone_utils::logInfo("Da mo thanh cong! Resolution: " +
        to_string(frame_w) + "x" + to_string(frame_h) +
        " | FPS: " + to_string(static_cast<int>(fps)));

    return cap;    // Trả về VideoCapture đã mở thành công
}

// ==============================================================================
// CHẾ ĐỘ 1: ArUco Marker Detection
// Phát hiện ArUco marker từ camera, ước tính vị trí 3D, tính khoảng cách
// ==============================================================================
void runArucoMode() {
    drone_utils::logInfo("=== CHE DO ARUCO DETECTION ===");

    // Mở camera/video
    VideoCapture cap = openCamera();     // Gọi hàm mở camera
    if (!cap.isOpened()) return;         // Thoát nếu không mở được

    // Lấy kích thước frame để tạo camera matrix mặc định
    int frame_w = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));    // Chiều rộng
    int frame_h = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));   // Chiều cao

    // Tạo camera matrix mặc định (giả lập thông số camera)
    drone_utils::CameraCalibration calib = 
        drone_utils::getDefaultCameraMatrix(frame_w, frame_h);

    // Khởi tạo ArUco Detector với thông số camera
    ArucoDetector aruco(calib.cameraMatrix, calib.distCoeffs);

    // Khởi tạo FPS counter để đo tốc độ xử lý
    drone_utils::FPSCounter fps_counter;

    // Khởi tạo HUD overlay
    DroneHUD hud(frame_w, frame_h);

    // Biến điều khiển
    bool show_hud = true;          // Có hiển thị HUD hay không
    float marker_length = 0.05f;   // Kích thước thực tế của marker (5cm)

    cout << "\n  === PHIM TAT ===\n";
    cout << "  [q/ESC] Thoat | [d] Doi dictionary | [h] Bat/tat HUD\n";
    cout << "  [+/-] Tang/giam kich thuoc marker | [s] Chup anh\n\n";

    // --- Vòng lặp chính xử lý từng frame ---
    while (true) {
        Mat frame;                     // Biến lưu frame hiện tại
        cap >> frame;                  // Đọc 1 frame từ camera

        // Kiểm tra frame có hợp lệ không (video hết hoặc camera lỗi)
        if (frame.empty()) {
            drone_utils::logWarning("Frame rong, dang thu lai...");
            // Nếu là video file, quay lại đầu
            cap.set(CAP_PROP_POS_FRAMES, 0);    // Đặt vị trí về frame 0
            continue;                             // Tiếp tục vòng lặp
        }

        // Cập nhật FPS counter
        fps_counter.tick();                       // Đánh dấu thời điểm frame mới

        // --- Phát hiện ArUco marker trong frame ---
        aruco.detect(frame);                      // Tìm tất cả ArUco marker

        // --- Ước tính pose 3D cho các marker đã phát hiện ---
        aruco.estimatePose(marker_length);         // Tính rotation & translation vectors

        // --- Tính khoảng cách từ camera đến từng marker ---
        aruco.calculateDistance(marker_length);     // Tính distance từ tvec

        // --- Vẽ kết quả phát hiện lên frame ---
        aruco.drawDetections(frame);               // Vẽ marker, ID, trục 3D, khoảng cách

        // --- Vẽ HUD overlay nếu được bật ---
        if (show_hud) {
            hud.setFPS(fps_counter.getFPS());      // Cập nhật FPS cho HUD

            // Nếu phát hiện landing pad, cập nhật altitude
            if (aruco.isLandingPadDetected()) {
                // Lấy khoảng cách của landing pad làm altitude
                auto markers = aruco.getDetections();    // Lấy danh sách marker
                for (const auto& m : markers) {
                    if (m.id == 0) {                      // Marker ID=0 là landing pad
                        hud.setAltitude(m.distance);      // Cập nhật cao độ
                        break;
                    }
                }
                hud.setFlightMode("LANDING");             // Đổi mode thành LANDING
            } else {
                hud.setFlightMode("STABILIZE");           // Mode mặc định
            }

            hud.drawHUD(frame);                    // Vẽ toàn bộ HUD lên frame
        }

        // --- Hiển thị frame kết quả ---
        imshow("DroneVision - ArUco Detection", frame);    // Hiện cửa sổ

        // --- Xử lý phím bấm ---
        int key = waitKey(1) & 0xFF;    // Đợi 1ms và đọc phím (mask 8 bit)

        if (key == 'q' || key == 27) {           // Phím 'q' hoặc ESC
            break;                                // Thoát vòng lặp
        }
        else if (key == 'h') {                   // Phím 'h'
            show_hud = !show_hud;                // Bật/tắt HUD
            drone_utils::logInfo(show_hud ? "HUD: BAT" : "HUD: TAT");
        }
        else if (key == 'd') {                   // Phím 'd' - đổi dictionary
            // Chuyển qua dictionary tiếp theo
            static int dict_idx = 0;              // Biến static giữ trạng thái
            // Danh sách các dictionary phổ biến
            int dicts[] = {
                cv::aruco::DICT_4X4_50,           // 4x4 bits, 50 markers
                cv::aruco::DICT_5X5_100,          // 5x5 bits, 100 markers
                cv::aruco::DICT_6X6_250,          // 6x6 bits, 250 markers
                cv::aruco::DICT_7X7_1000,         // 7x7 bits, 1000 markers
                cv::aruco::DICT_ARUCO_ORIGINAL    // ArUco gốc
            };
            string dict_names[] = {"4x4_50", "5x5_100", "6x6_250", "7x7_1000", "ORIGINAL"};
            dict_idx = (dict_idx + 1) % 5;       // Chuyển sang dictionary kế tiếp
            aruco.setDictionary(dicts[dict_idx]); // Áp dụng dictionary mới
            drone_utils::logInfo("Dictionary: " + dict_names[dict_idx]);
        }
        else if (key == '+' || key == '=') {     // Phím '+' tăng kích thước marker
            marker_length += 0.01f;               // Tăng 1cm
            drone_utils::logInfo("Marker size: " + to_string(marker_length) + "m");
        }
        else if (key == '-') {                    // Phím '-' giảm kích thước marker
            marker_length = max(0.01f, marker_length - 0.01f);    // Giảm 1cm (tối thiểu 1cm)
            drone_utils::logInfo("Marker size: " + to_string(marker_length) + "m");
        }
        else if (key == 's') {                   // Phím 's' chụp screenshot
            string filename = "screenshot_aruco_" + drone_utils::getCurrentTimestamp() + ".png";
            imwrite(filename, frame);             // Lưu frame ra file PNG
            drone_utils::logInfo("Da luu: " + filename);
        }
    }

    // Giải phóng tài nguyên
    cap.release();                    // Đóng camera/video
    destroyAllWindows();              // Đóng tất cả cửa sổ OpenCV
    drone_utils::logInfo("Da thoat che do ArUco Detection");
}

// ==============================================================================
// CHẾ ĐỘ 2: QR Code Scanner
// Quét và giải mã QR code thời gian thực, phân tích lệnh drone
// ==============================================================================
void runQRMode() {
    drone_utils::logInfo("=== CHE DO QR SCANNER ===");

    // Mở camera/video
    VideoCapture cap = openCamera();
    if (!cap.isOpened()) return;

    // Lấy kích thước frame
    int frame_w = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // Khởi tạo QR Detector
    QRDetector qr;

    // Khởi tạo FPS counter và HUD
    drone_utils::FPSCounter fps_counter;
    DroneHUD hud(frame_w, frame_h);

    bool show_hud = true;    // Bật HUD mặc định

    cout << "\n  === PHIM TAT ===\n";
    cout << "  [q/ESC] Thoat | [h] Bat/tat HUD | [c] Xoa lich su\n";
    cout << "  [s] Chup anh | [l] Xem lich su quet\n\n";

    // --- Vòng lặp chính ---
    while (true) {
        Mat frame;
        cap >> frame;                 // Đọc frame

        if (frame.empty()) {
            cap.set(CAP_PROP_POS_FRAMES, 0);    // Reset video nếu hết
            continue;
        }

        fps_counter.tick();           // Cập nhật FPS

        // --- Phát hiện và giải mã QR code ---
        qr.detect(frame);            // Tìm QR code trong frame

        // --- Vẽ kết quả phát hiện ---
        qr.drawDetections(frame);    // Vẽ bounding box và nội dung QR

        // --- Phân tích lệnh drone từ QR code ---
        auto decoded = qr.getDecodedData();    // Lấy danh sách QR đã giải mã
        for (const auto& qr_info : decoded) {
            // Phân tích nội dung QR thành lệnh drone
            auto cmd = qr.parseCommand(qr_info.data);
            if (!cmd.type.empty()) {
                // Hiển thị lệnh đã phân tích lên frame
                string cmd_text = "CMD: " + cmd.type;
                for (const auto& p : cmd.params) {
                    cmd_text += " | " + p.first + "=" + p.second;
                }
                // Vẽ text lệnh phía trên frame
                drone_utils::drawTextWithBackground(frame, cmd_text,
                    Point(10, frame.rows - 40),    // Vị trí: góc dưới trái
                    FONT_HERSHEY_SIMPLEX,          // Font chữ
                    0.6,                            // Kích thước font
                    Scalar(0, 255, 255),            // Màu vàng (BGR)
                    Scalar(0, 0, 0),                // Nền đen
                    1);                             // Độ dày chữ
            }
        }

        // --- Vẽ HUD ---
        if (show_hud) {
            hud.setFPS(fps_counter.getFPS());
            hud.setFlightMode("QR_SCAN");
            hud.setDetectionCount(static_cast<int>(decoded.size()));
            hud.drawHUD(frame);
        }

        // --- Hiển thị ---
        imshow("DroneVision - QR Scanner", frame);

        // --- Xử lý phím ---
        int key = waitKey(1) & 0xFF;

        if (key == 'q' || key == 27) break;          // Thoát
        else if (key == 'h') {
            show_hud = !show_hud;                      // Bật/tắt HUD
        }
        else if (key == 'c') {
            qr.clearHistory();                         // Xóa lịch sử QR
            drone_utils::logInfo("Da xoa lich su QR");
        }
        else if (key == 'l') {
            // Hiển thị lịch sử QR đã quét ra console
            auto history = qr.getHistory();
            cout << "\n  === LICH SU QR DA QUET ===\n";
            int idx = 1;
            for (const auto& item : history) {
                cout << "  [" << idx++ << "] " << item.data << "\n";
            }
            if (history.empty()) cout << "  (Chua quet duoc QR nao)\n";
            cout << "  ========================\n\n";
        }
        else if (key == 's') {
            string filename = "screenshot_qr_" + drone_utils::getCurrentTimestamp() + ".png";
            imwrite(filename, frame);
            drone_utils::logInfo("Da luu: " + filename);
        }
    }

    cap.release();
    destroyAllWindows();
    drone_utils::logInfo("Da thoat che do QR Scanner");
}

// ==============================================================================
// CHẾ ĐỘ 3: Object Detection (YOLO)
// Phát hiện đối tượng sử dụng YOLO qua OpenCV DNN module
// ==============================================================================
void runObjectDetectionMode() {
    drone_utils::logInfo("=== CHE DO OBJECT DETECTION ===");

    // Đường dẫn đến các file model YOLO
    string cfg_path = "assets/models/yolov4-tiny.cfg";          // File cấu hình mạng
    string weights_path = "assets/models/yolov4-tiny.weights";  // File trọng số đã train
    string names_path = "assets/models/coco.names";             // File tên các class

    // Kiểm tra file model có tồn tại không
    if (!fs::exists(cfg_path) || !fs::exists(weights_path) || !fs::exists(names_path)) {
        drone_utils::logError("Khong tim thay file model YOLO!");
        cout << "\n  Vui long tai cac file sau vao thu muc assets/models/:\n";
        cout << "  1. yolov4-tiny.cfg    - https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg\n";
        cout << "  2. yolov4-tiny.weights - https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights\n";
        cout << "  3. coco.names         - https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names\n";
        cout << "\n  Nhan Enter de quay lai menu...";
        cin.ignore();
        cin.get();
        return;
    }

    // Khởi tạo Object Detector với model YOLO
    ObjectDetector detector(cfg_path, weights_path, names_path);

    // Kiểm tra model đã load thành công
    if (!detector.isModelLoaded()) {
        drone_utils::logError("Khong the load model YOLO!");
        return;
    }

    drone_utils::logInfo("Da load model YOLO thanh cong!");

    // Mở camera
    VideoCapture cap = openCamera();
    if (!cap.isOpened()) return;

    int frame_w = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // Khởi tạo FPS counter và HUD
    drone_utils::FPSCounter fps_counter;
    DroneHUD hud(frame_w, frame_h);

    // Tham số phát hiện
    float conf_threshold = 0.5f;     // Ngưỡng confidence (50%)
    float nms_threshold = 0.4f;      // Ngưỡng NMS (Non-Maximum Suppression)
    bool show_hud = true;
    bool use_roi = false;            // Có sử dụng ROI detection không
    Rect roi_rect;                   // Vùng ROI

    cout << "\n  === PHIM TAT ===\n";
    cout << "  [q/ESC] Thoat | [h] Bat/tat HUD\n";
    cout << "  [+/-] Tang/giam confidence | [r] Bat/tat ROI\n";
    cout << "  [s] Chup anh | [c] Dem doi tuong\n\n";

    // --- Vòng lặp chính ---
    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        fps_counter.tick();

        // --- Phát hiện đối tượng ---
        if (use_roi && !roi_rect.empty()) {
            // Chỉ phát hiện trong vùng ROI (tiết kiệm tài nguyên)
            detector.detectInROI(frame, roi_rect);
        } else {
            // Phát hiện toàn bộ frame
            detector.detect(frame, conf_threshold, nms_threshold);
        }

        // --- Vẽ kết quả phát hiện ---
        detector.drawDetections(frame);

        // Vẽ ROI nếu đang sử dụng
        if (use_roi && !roi_rect.empty()) {
            rectangle(frame, roi_rect, Scalar(255, 255, 0), 2);    // Vẽ khung ROI màu cyan
            putText(frame, "ROI", Point(roi_rect.x, roi_rect.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
        }

        // --- Hiển thị thông tin confidence threshold ---
        string conf_text = "Confidence: " + to_string(static_cast<int>(conf_threshold * 100)) + "%";
        drone_utils::drawTextWithBackground(frame, conf_text,
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(255, 255, 255), Scalar(50, 50, 50), 1);

        // --- Vẽ HUD ---
        if (show_hud) {
            hud.setFPS(fps_counter.getFPS());
            hud.setFlightMode("DETECT");
            hud.setDetectionCount(static_cast<int>(detector.getDetections().size()));
            hud.drawHUD(frame);
        }

        imshow("DroneVision - Object Detection", frame);

        // --- Xử lý phím ---
        int key = waitKey(1) & 0xFF;

        if (key == 'q' || key == 27) break;
        else if (key == 'h') show_hud = !show_hud;
        else if (key == '+' || key == '=') {
            conf_threshold = min(0.95f, conf_threshold + 0.05f);    // Tăng 5%
            drone_utils::logInfo("Confidence threshold: " +
                to_string(static_cast<int>(conf_threshold * 100)) + "%");
        }
        else if (key == '-') {
            conf_threshold = max(0.1f, conf_threshold - 0.05f);     // Giảm 5%
            drone_utils::logInfo("Confidence threshold: " +
                to_string(static_cast<int>(conf_threshold * 100)) + "%");
        }
        else if (key == 'r') {
            if (!use_roi) {
                // Cho phép người dùng vẽ ROI bằng chuột
                drone_utils::logInfo("Chon vung ROI tren anh (keo chuot va nhan Enter)...");
                roi_rect = selectROI("DroneVision - Object Detection", frame, false);
                if (roi_rect.width > 0 && roi_rect.height > 0) {
                    use_roi = true;
                    drone_utils::logInfo("ROI da duoc thiet lap");
                }
            } else {
                use_roi = false;      // Tắt ROI
                drone_utils::logInfo("ROI da tat");
            }
        }
        else if (key == 'c') {
            // Hiển thị đếm đối tượng theo class
            auto counts = detector.countByClass();
            cout << "\n  === DEM DOI TUONG ===\n";
            for (const auto& pair : counts) {
                cout << "  " << pair.first << ": " << pair.second << "\n";
            }
            cout << "  ====================\n\n";
        }
        else if (key == 's') {
            string filename = "screenshot_detect_" + drone_utils::getCurrentTimestamp() + ".png";
            imwrite(filename, frame);
            drone_utils::logInfo("Da luu: " + filename);
        }
    }

    cap.release();
    destroyAllWindows();
    drone_utils::logInfo("Da thoat che do Object Detection");
}

// ==============================================================================
// CHẾ ĐỘ 4: Full Mission Mode
// Kết hợp tất cả module + HUD overlay
// ==============================================================================
void runFullMissionMode() {
    drone_utils::logInfo("=== CHE DO FULL MISSION ===");

    // Mở camera
    VideoCapture cap = openCamera();
    if (!cap.isOpened()) return;

    int frame_w = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // --- Khởi tạo tất cả module ---
    // Camera calibration mặc định
    drone_utils::CameraCalibration calib =
        drone_utils::getDefaultCameraMatrix(frame_w, frame_h);

    // ArUco Detector
    ArucoDetector aruco(calib.cameraMatrix, calib.distCoeffs);

    // QR Detector
    QRDetector qr;

    // Object Detector (chỉ load nếu có model)
    bool has_yolo = false;                   // Cờ đánh dấu có YOLO model không
    ObjectDetector* detector = nullptr;       // Con trỏ đến Object Detector

    // Kiểm tra và load YOLO model nếu có
    if (fs::exists("assets/models/yolov4-tiny.cfg") &&
        fs::exists("assets/models/yolov4-tiny.weights") &&
        fs::exists("assets/models/coco.names")) {
        detector = new ObjectDetector(
            "assets/models/yolov4-tiny.cfg",
            "assets/models/yolov4-tiny.weights",
            "assets/models/coco.names"
        );
        has_yolo = detector->isModelLoaded();
        if (has_yolo) {
            drone_utils::logInfo("YOLO model loaded thanh cong!");
        }
    }
    if (!has_yolo) {
        drone_utils::logWarning("Khong co YOLO model - Object Detection se bi tat");
    }

    // Image Processor
    ImageProcessor imgproc;

    // HUD
    DroneHUD hud(frame_w, frame_h);

    // FPS Counter
    drone_utils::FPSCounter fps_counter;

    // --- Cờ bật/tắt từng module ---
    bool enable_aruco = true;        // Bật ArUco detection
    bool enable_qr = true;           // Bật QR detection
    bool enable_yolo = has_yolo;     // Bật YOLO (nếu có model)
    bool enable_imgproc = false;     // Tắt image processing mặc định
    bool enable_hud = true;          // Bật HUD
    int imgproc_mode = 0;            // Chế độ xử lý ảnh (0=none, 1=CLAHE, 2=Dehaze, 3=Edge)

    // Giả lập telemetry data
    double sim_heading = 0.0;        // Hướng la bàn giả lập
    int sim_battery = 85;            // Pin giả lập (%)
    double sim_lat = 21.0285;        // Vĩ độ giả lập (Hà Nội)
    double sim_lon = 105.8542;       // Kinh độ giả lập (Hà Nội)

    cout << "\n  === PHIM TAT FULL MISSION ===\n";
    cout << "  [1] Bat/tat ArUco  [2] Bat/tat QR  [3] Bat/tat YOLO\n";
    cout << "  [4] Doi filter anh [5] Bat/tat HUD  [s] Chup anh\n";
    cout << "  [q/ESC] Thoat\n\n";

    // --- Vòng lặp chính ---
    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        fps_counter.tick();

        // --- Áp dụng xử lý ảnh nếu bật ---
        if (enable_imgproc) {
            switch (imgproc_mode) {
                case 1:
                    frame = imgproc.applyCLAHE(frame);          // Cân bằng histogram
                    break;
                case 2:
                    frame = imgproc.dehaze(frame);               // Khử sương mù
                    break;
                case 3:
                    frame = imgproc.detectEdges(frame, 0);       // Phát hiện cạnh Canny
                    break;
            }
        }

        // --- Phát hiện ArUco markers ---
        if (enable_aruco) {
            aruco.detect(frame);                    // Phát hiện marker
            aruco.estimatePose(0.05f);              // Ước tính pose (marker 5cm)
            aruco.calculateDistance(0.05f);          // Tính khoảng cách
            aruco.drawDetections(frame);            // Vẽ kết quả
        }

        // --- Phát hiện QR code ---
        if (enable_qr) {
            qr.detect(frame);                       // Phát hiện QR
            qr.drawDetections(frame);               // Vẽ kết quả
        }

        // --- Phát hiện đối tượng YOLO ---
        if (enable_yolo && detector != nullptr) {
            detector->detect(frame, 0.5f, 0.4f);    // Detect với conf=0.5
            detector->drawDetections(frame);          // Vẽ bounding boxes
        }

        // --- Cập nhật và vẽ HUD ---
        if (enable_hud) {
            hud.setFPS(fps_counter.getFPS());

            // Cập nhật altitude từ ArUco nếu phát hiện landing pad
            if (enable_aruco && aruco.isLandingPadDetected()) {
                auto markers = aruco.getDetections();
                for (const auto& m : markers) {
                    if (m.id == 0) {
                        hud.setAltitude(m.distance);
                        break;
                    }
                }
                hud.setFlightMode("LANDING");
            } else {
                hud.setFlightMode("MISSION");
            }

            // Giả lập la bàn quay (để demo)
            sim_heading = fmod(sim_heading + 0.5, 360.0);    // Quay 0.5 độ/frame
            hud.setHeading(sim_heading);
            hud.setBattery(sim_battery);
            hud.setGPSCoords(sim_lat, sim_lon);

            // Đếm tổng detection
            int total_detections = 0;
            if (enable_aruco) total_detections += static_cast<int>(aruco.getDetections().size());
            if (enable_qr) total_detections += static_cast<int>(qr.getDecodedData().size());
            if (enable_yolo && detector) total_detections += static_cast<int>(detector->getDetections().size());
            hud.setDetectionCount(total_detections);

            hud.drawHUD(frame);
        }

        // --- Hiển thị trạng thái module ở góc trên ---
        string status = "Modules: ";
        status += enable_aruco ? "[ArUco:ON] " : "[ArUco:OFF] ";
        status += enable_qr ? "[QR:ON] " : "[QR:OFF] ";
        status += enable_yolo ? "[YOLO:ON] " : "[YOLO:OFF] ";
        status += enable_imgproc ? "[Filter:" + to_string(imgproc_mode) + "] " : "[Filter:OFF] ";

        drone_utils::drawTextWithBackground(frame, status,
            Point(10, frame.rows - 15), FONT_HERSHEY_SIMPLEX, 0.45,
            Scalar(200, 200, 200), Scalar(30, 30, 30), 1);

        imshow("DroneVision - Full Mission", frame);

        // --- Xử lý phím ---
        int key = waitKey(1) & 0xFF;

        if (key == 'q' || key == 27) break;
        else if (key == '1') {
            enable_aruco = !enable_aruco;
            drone_utils::logInfo(string("ArUco: ") + (enable_aruco ? "BAT" : "TAT"));
        }
        else if (key == '2') {
            enable_qr = !enable_qr;
            drone_utils::logInfo(string("QR: ") + (enable_qr ? "BAT" : "TAT"));
        }
        else if (key == '3') {
            if (has_yolo) {
                enable_yolo = !enable_yolo;
                drone_utils::logInfo(string("YOLO: ") + (enable_yolo ? "BAT" : "TAT"));
            } else {
                drone_utils::logWarning("Khong co YOLO model!");
            }
        }
        else if (key == '4') {
            enable_imgproc = true;
            imgproc_mode = (imgproc_mode + 1) % 4;    // Chuyển filter: 0->1->2->3->0
            string modes[] = {"None", "CLAHE", "Dehaze", "Edge"};
            if (imgproc_mode == 0) enable_imgproc = false;
            drone_utils::logInfo("Image Filter: " + modes[imgproc_mode]);
        }
        else if (key == '5') {
            enable_hud = !enable_hud;
            drone_utils::logInfo(string("HUD: ") + (enable_hud ? "BAT" : "TAT"));
        }
        else if (key == 's') {
            string filename = "screenshot_mission_" + drone_utils::getCurrentTimestamp() + ".png";
            imwrite(filename, frame);
            drone_utils::logInfo("Da luu: " + filename);
        }
    }

    // Giải phóng tài nguyên
    if (detector != nullptr) {
        delete detector;        // Xóa Object Detector
        detector = nullptr;
    }
    cap.release();
    destroyAllWindows();
    drone_utils::logInfo("Da thoat che do Full Mission");
}

// ==============================================================================
// CHẾ ĐỘ 5: Image Processing Lab
// Thử nghiệm các bộ lọc xử lý ảnh
// ==============================================================================
void runImageProcessingLab() {
    drone_utils::logInfo("=== CHE DO IMAGE PROCESSING LAB ===");

    // Mở camera
    VideoCapture cap = openCamera();
    if (!cap.isOpened()) return;

    // Khởi tạo Image Processor
    ImageProcessor imgproc;
    drone_utils::FPSCounter fps_counter;

    // Biến điều khiển
    int current_filter = 0;             // Filter hiện tại (0 = original)
    float zoom_factor = 1.0f;           // Mức zoom
    int edge_method = 0;                // Phương pháp edge (0=Canny, 1=Sobel, 2=Laplacian)
    double brightness_alpha = 1.0;      // Hệ số brightness
    int brightness_beta = 0;            // Offset brightness

    // Danh sách các filter
    string filter_names[] = {
        "Original",                      // 0: Ảnh gốc
        "CLAHE (Histogram)",             // 1: Cân bằng histogram
        "Dehaze (Khu suong)",            // 2: Khử sương mù
        "Edge Detection",               // 3: Phát hiện cạnh
        "Color Filter (Red)",            // 4: Lọc màu đỏ
        "Color Filter (Green)",          // 5: Lọc màu xanh lá
        "Color Filter (Blue)",           // 6: Lọc màu xanh dương
        "Morphology (Open)",             // 7: Morphological Opening
        "Morphology (Close)",            // 8: Morphological Closing
        "Sharpen",                       // 9: Làm nét
        "Denoise",                       // 10: Khử nhiễu
        "Brightness/Contrast"            // 11: Điều chỉnh sáng/tương phản
    };
    int total_filters = 12;              // Tổng số filter

    cout << "\n  === PHIM TAT IMAGE LAB ===\n";
    cout << "  [LEFT/RIGHT] Chuyen filter | [UP/DOWN] Zoom in/out\n";
    cout << "  [e] Doi edge method | [s] Chup anh | [q/ESC] Thoat\n\n";

    // --- Vòng lặp chính ---
    while (true) {
        Mat frame, result;
        cap >> frame;

        if (frame.empty()) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        fps_counter.tick();

        // --- Áp dụng filter theo lựa chọn ---
        switch (current_filter) {
            case 0:     // Original - không xử lý
                result = frame.clone();
                break;
            case 1:     // CLAHE
                result = imgproc.applyCLAHE(frame, 2.0, Size(8, 8));
                break;
            case 2:     // Dehaze
                result = imgproc.dehaze(frame, 0.7f);
                break;
            case 3:     // Edge Detection
                result = imgproc.detectEdges(frame, edge_method);
                break;
            case 4:     // Color Filter - Red
                result = imgproc.filterByColor(frame,
                    Scalar(0, 100, 100),       // HSV lower bound (đỏ)
                    Scalar(10, 255, 255));      // HSV upper bound (đỏ)
                break;
            case 5:     // Color Filter - Green
                result = imgproc.filterByColor(frame,
                    Scalar(35, 100, 100),       // HSV lower bound (xanh lá)
                    Scalar(85, 255, 255));       // HSV upper bound (xanh lá)
                break;
            case 6:     // Color Filter - Blue
                result = imgproc.filterByColor(frame,
                    Scalar(100, 100, 100),      // HSV lower bound (xanh dương)
                    Scalar(130, 255, 255));      // HSV upper bound (xanh dương)
                break;
            case 7:     // Morphology Open
                result = imgproc.applyMorphology(frame, 2, 5);    // MORPH_OPEN
                break;
            case 8:     // Morphology Close
                result = imgproc.applyMorphology(frame, 3, 5);    // MORPH_CLOSE
                break;
            case 9:     // Sharpen
                result = imgproc.sharpen(frame);
                break;
            case 10:    // Denoise
                result = imgproc.denoise(frame);
                break;
            case 11:    // Brightness/Contrast
                result = imgproc.adjustBrightnessContrast(frame, brightness_alpha, brightness_beta);
                break;
            default:
                result = frame.clone();
                break;
        }

        // --- Áp dụng Digital Zoom ---
        if (zoom_factor > 1.0f) {
            result = imgproc.digitalZoom(result, zoom_factor);
        }

        // --- Hiển thị thông tin filter ---
        string info = "Filter: " + filter_names[current_filter] +
                      " | FPS: " + to_string(static_cast<int>(fps_counter.getFPS())) +
                      " | Zoom: " + to_string(zoom_factor).substr(0, 3) + "x";
        drone_utils::drawTextWithBackground(result, info,
            Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(0, 255, 0), Scalar(0, 0, 0), 1);

        // Hiển thị thêm thông tin cho một số filter
        if (current_filter == 3) {
            string edge_names[] = {"Canny", "Sobel", "Laplacian"};
            drone_utils::drawTextWithBackground(result, "Edge: " + edge_names[edge_method],
                Point(10, 55), FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(255, 255, 0), Scalar(0, 0, 0), 1);
        }
        if (current_filter == 11) {
            string bc_info = "Alpha: " + to_string(brightness_alpha).substr(0, 3) +
                             " | Beta: " + to_string(brightness_beta);
            drone_utils::drawTextWithBackground(result, bc_info,
                Point(10, 55), FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(255, 255, 0), Scalar(0, 0, 0), 1);
        }

        imshow("DroneVision - Image Processing Lab", result);

        // --- Xử lý phím ---
        int key = waitKey(1) & 0xFF;

        if (key == 'q' || key == 27) break;
        else if (key == 83 || key == 'd') {      // RIGHT arrow hoặc 'd'
            current_filter = (current_filter + 1) % total_filters;    // Filter tiếp
            drone_utils::logInfo("Filter: " + filter_names[current_filter]);
        }
        else if (key == 81 || key == 'a') {      // LEFT arrow hoặc 'a'
            current_filter = (current_filter - 1 + total_filters) % total_filters;    // Filter trước
            drone_utils::logInfo("Filter: " + filter_names[current_filter]);
        }
        else if (key == 82 || key == 'w') {      // UP arrow hoặc 'w' - Zoom in
            zoom_factor = min(5.0f, zoom_factor + 0.1f);
        }
        else if (key == 84 || key == 'x') {      // DOWN arrow hoặc 'x' - Zoom out
            zoom_factor = max(1.0f, zoom_factor - 0.1f);
        }
        else if (key == 'e') {                    // Đổi edge method
            edge_method = (edge_method + 1) % 3;
            string edge_names[] = {"Canny", "Sobel", "Laplacian"};
            drone_utils::logInfo("Edge method: " + edge_names[edge_method]);
        }
        else if (key == 'b') {                    // Tăng brightness
            brightness_beta = min(100, brightness_beta + 10);
        }
        else if (key == 'n') {                    // Giảm brightness
            brightness_beta = max(-100, brightness_beta - 10);
        }
        else if (key == 's') {
            string filename = "screenshot_lab_" + drone_utils::getCurrentTimestamp() + ".png";
            imwrite(filename, result);
            drone_utils::logInfo("Da luu: " + filename);
        }
    }

    cap.release();
    destroyAllWindows();
    drone_utils::logInfo("Da thoat che do Image Processing Lab");
}

// ==============================================================================
// CHẾ ĐỘ 6: ArUco Marker Generator
// Tạo ảnh ArUco marker để in ra giấy
// ==============================================================================
void runArucoGenerator() {
    drone_utils::logInfo("=== CHE DO ARUCO GENERATOR ===");

    // Tạo thư mục output nếu chưa có
    fs::create_directories("assets/generated_markers");

    ArucoDetector aruco;    // Khởi tạo ArUco Detector (dùng hàm generateMarker)

    // Hỏi người dùng tham số
    cout << "\n  === TAO ARUCO MARKER ===\n\n";

    // Chọn dictionary
    cout << "  Chon dictionary:\n";
    cout << "  [1] DICT_4X4_50    (4x4 bits, 50 markers)\n";
    cout << "  [2] DICT_5X5_100   (5x5 bits, 100 markers)\n";
    cout << "  [3] DICT_6X6_250   (6x6 bits, 250 markers)\n";
    cout << "  [4] DICT_7X7_1000  (7x7 bits, 1000 markers)\n";
    cout << "  [5] DICT_ARUCO_ORIGINAL\n";
    cout << "  >> Lua chon (1-5): ";

    int dict_choice;
    cin >> dict_choice;

    // Map lựa chọn sang OpenCV dictionary type
    int dict_type;
    switch (dict_choice) {
        case 1: dict_type = cv::aruco::DICT_4X4_50; break;
        case 2: dict_type = cv::aruco::DICT_5X5_100; break;
        case 3: dict_type = cv::aruco::DICT_6X6_250; break;
        case 4: dict_type = cv::aruco::DICT_7X7_1000; break;
        case 5: dict_type = cv::aruco::DICT_ARUCO_ORIGINAL; break;
        default: dict_type = cv::aruco::DICT_4X4_50; break;
    }

    // Nhập ID marker
    cout << "  Nhap ID marker (0-49 cho 4x4_50): ";
    int marker_id;
    cin >> marker_id;

    // Nhập kích thước ảnh (pixels)
    cout << "  Nhap kich thuoc anh (pixels, vi du 200): ";
    int marker_size;
    cin >> marker_size;

    if (marker_size < 50) marker_size = 200;    // Mặc định 200 nếu quá nhỏ

    // --- Tạo marker ---
    Mat marker_img = aruco.generateMarker(marker_id, marker_size, dict_type);

    if (marker_img.empty()) {
        drone_utils::logError("Khong the tao marker!");
        return;
    }

    // Tạo tên file với thông tin chi tiết
    string filename = "assets/generated_markers/aruco_dict" +
                      to_string(dict_choice) + "_id" +
                      to_string(marker_id) + "_" +
                      to_string(marker_size) + "px.png";

    // Lưu ảnh marker
    imwrite(filename, marker_img);
    drone_utils::logInfo("Da tao marker: " + filename);

    // Hiển thị marker
    imshow("Generated ArUco Marker", marker_img);
    cout << "\n  Marker da duoc luu tai: " << filename << "\n";
    cout << "  Nhan phim bat ky de tiep tuc...\n";
    waitKey(0);              // Đợi người dùng nhấn phím

    // Hỏi có muốn tạo batch markers không
    cout << "\n  Ban co muon tao nhieu marker cung luc? (y/n): ";
    char batch_choice;
    cin >> batch_choice;

    if (batch_choice == 'y' || batch_choice == 'Y') {
        cout << "  Nhap so luong marker can tao: ";
        int count;
        cin >> count;

        // Tạo batch markers
        for (int i = 0; i < count; i++) {
            Mat batch_marker = aruco.generateMarker(i, marker_size, dict_type);
            string batch_filename = "assets/generated_markers/aruco_dict" +
                                    to_string(dict_choice) + "_id" +
                                    to_string(i) + "_" +
                                    to_string(marker_size) + "px.png";
            imwrite(batch_filename, batch_marker);
            cout << "  Da tao: " << batch_filename << "\n";
        }
        drone_utils::logInfo("Da tao " + to_string(count) + " markers!");
    }

    destroyAllWindows();
}

// ==============================================================================
// HÀM MAIN - Entry point của chương trình
// ==============================================================================
int main(int argc, char** argv) {
    // In thông tin OpenCV version
    drone_utils::logInfo("OpenCV version: " + string(CV_VERSION));
    drone_utils::logInfo("DroneVisionPro v1.0 - Khoi dong thanh cong!");

    // --- Vòng lặp menu chính ---
    while (true) {
        showMainMenu();          // Hiển thị menu

        int choice;              // Biến lưu lựa chọn
        cin >> choice;           // Đọc lựa chọn từ bàn phím

        // Xử lý input không hợp lệ
        if (cin.fail()) {
            cin.clear();                                  // Xóa trạng thái lỗi
            cin.ignore(numeric_limits<streamsize>::max(), '\n');  // Xóa buffer
            drone_utils::logWarning("Nhap khong hop le!");
            continue;                                     // Quay lại menu
        }

        // --- Chuyển đến chế độ tương ứng ---
        switch (choice) {
            case 0:
                // Thoát chương trình
                drone_utils::logInfo("Tam biet! Hen gap lai!");
                cout << "\n  === CHUONG TRINH DA DONG ===\n\n";
                return 0;        // Kết thúc chương trình với mã 0 (thành công)

            case 1:
                // Chế độ ArUco Detection
                runArucoMode();
                break;

            case 2:
                // Chế độ QR Scanner
                runQRMode();
                break;

            case 3:
                // Chế độ Object Detection
                runObjectDetectionMode();
                break;

            case 4:
                // Chế độ Full Mission (tất cả module + HUD)
                runFullMissionMode();
                break;

            case 5:
                // Chế độ Image Processing Lab
                runImageProcessingLab();
                break;

            case 6:
                // Chế độ ArUco Generator
                runArucoGenerator();
                break;

            default:
                // Lựa chọn không hợp lệ
                drone_utils::logWarning("Lua chon khong hop le! Vui long chon 0-6.");
                break;
        }
    }

    return 0;    // Kết thúc chương trình (không bao giờ đến đây do return trong case 0)
}
