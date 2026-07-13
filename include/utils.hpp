#pragma once // Chỉ thị tiền xử lý đảm bảo file header chỉ được include một lần, tránh lỗi định nghĩa trùng lặp

// ============================================================================
// FILE: include/utils.hpp
// PROJECT: DroneVisionPro - Hệ thống thị giác máy tính cho drone
// PURPOSE: Các tiện ích chung - đếm FPS, hiệu chỉnh camera, hằng số màu, logging
// AUTHOR: Tran Ngoc Bao - 24021238
// DATE: 2026-07-13
// DESCRIPTION: Cung cấp các hàm và lớp tiện ích dùng chung cho toàn bộ dự án
//              bao gồm: FPSCounter, CameraCalibration, màu sắc, logging, vẽ text
// ============================================================================

#include <opencv2/opencv.hpp>    // Thư viện OpenCV chính - cung cấp Mat, Scalar, Point và các hàm vẽ
#include <string>                // Thư viện chuỗi C++ chuẩn - dùng cho tên file, thông điệp log
#include <chrono>                // Thư viện thời gian C++ - dùng cho FPSCounter và timestamp
#include <deque>                 // Thư viện hàng đợi hai đầu - dùng làm cửa sổ trượt trong FPSCounter
#include <iostream>              // Thư viện I/O chuẩn - dùng cho logging ra console
#include <iomanip>               // Thư viện định dạng I/O - dùng để format timestamp với setw, setfill
#include <sstream>               // Thư viện string stream - dùng để tạo chuỗi timestamp định dạng
#include <ctime>                 // Thư viện thời gian C - dùng cho localtime để lấy giờ hệ thống
#include <filesystem>            // Thư viện hệ thống file C++17 - dùng để tạo thư mục

/// @brief Namespace chứa tất cả các tiện ích chung của dự án DroneVisionPro
/// Namespace này tổ chức code gọn gàng, tránh xung đột tên với thư viện khác
namespace drone_utils {

// ============================================================================
// LỚP FPSCounter - BỘ ĐẾM SỐ KHUNG HÌNH TRÊN GIÂY
// ============================================================================

/// @brief Lớp FPSCounter - Đo FPS chính xác bằng cửa sổ trượt thời gian
/// Sử dụng kỹ thuật sliding window: lưu N timestamp gần nhất và tính FPS từ chênh lệch thời gian
/// Phương pháp này cho kết quả ổn định hơn so với đo từng frame đơn lẻ
class FPSCounter {
public:
    /// @brief Constructor khởi tạo bộ đếm FPS
    /// @param window_size Kích thước cửa sổ trượt - số lượng frame dùng để tính trung bình FPS
    /// Giá trị mặc định 30 cho kết quả mượt mà, không dao động quá nhiều
    explicit FPSCounter(size_t window_size = 30);

    /// @brief Destructor mặc định - không cần giải phóng tài nguyên đặc biệt
    ~FPSCounter() = default;

    /// @brief Ghi nhận một tick (một khung hình đã được xử lý)
    /// Gọi phương thức này mỗi khi hoàn thành xử lý một frame để đo FPS chính xác
    void tick();

    /// @brief Lấy giá trị FPS hiện tại đã được tính toán
    /// @return FPS trung bình dựa trên cửa sổ trượt - trả về 0.0 nếu chưa đủ 2 mẫu
    double getFPS() const;

private:
    /// @brief Hàng đợi hai đầu lưu các mốc thời gian của từng frame
    /// Deque cho phép thêm ở cuối và xóa ở đầu với độ phức tạp O(1)
    std::deque<std::chrono::steady_clock::time_point> timestamps_;

    /// @brief Kích thước tối đa của cửa sổ trượt
    /// Khi deque đầy, timestamp cũ nhất sẽ bị loại bỏ để nhường chỗ cho timestamp mới
    size_t window_size_;
};

// ============================================================================
// CẤU TRÚC CameraCalibration - THÔNG SỐ HIỆU CHỈNH CAMERA
// ============================================================================

/// @brief Cấu trúc lưu trữ thông số hiệu chỉnh camera (camera calibration)
/// Các thông số này dùng để chuyển đổi giữa tọa độ ảnh và tọa độ thực tế 3D
/// Cần thiết cho việc ước tính vị trí ArUco marker và các phép đo khoảng cách
struct CameraCalibration {
    cv::Mat camera_matrix;   // Ma trận camera nội tại 3x3 - chứa focal length (fx,fy) và điểm chính (cx,cy)
    cv::Mat dist_coeffs;     // Vector hệ số méo (distortion coefficients) - 5 phần tử (k1,k2,p1,p2,k3)
    bool is_calibrated;      // Cờ cho biết camera đã được hiệu chỉnh chính xác hay chưa

    /// @brief Constructor mặc định - khởi tạo trạng thái chưa hiệu chỉnh
    CameraCalibration() : is_calibrated(false) {} // Ban đầu camera chưa được hiệu chỉnh
};

// ============================================================================
// HÀM TIỆN ÍCH CAMERA
// ============================================================================

/// @brief Tạo ma trận camera nội tại mặc định với các giá trị hợp lý
/// Ma trận mặc định sử dụng focal length = chiều rộng frame, điểm chính ở tâm frame
/// Phù hợp cho camera góc rộng thông thường trên drone (FOV ~60-70 độ)
/// @param frame_width Chiều rộng khung hình tính bằng pixel
/// @param frame_height Chiều cao khung hình tính bằng pixel
/// @return Cấu trúc CameraCalibration với ma trận mặc định và distortion = 0
CameraCalibration getDefaultCameraMatrix(int frame_width, int frame_height);

// ============================================================================
// HÀM TIỆN ÍCH HỆ THỐNG FILE
// ============================================================================

/// @brief Tạo thư mục nếu chưa tồn tại, bao gồm cả các thư mục cha
/// Sử dụng std::filesystem::create_directories để tạo đệ quy
/// @param path Đường dẫn thư mục cần tạo - có thể là đường dẫn tuyệt đối hoặc tương đối
/// @return true nếu thư mục đã tồn tại hoặc tạo thành công, false nếu thất bại
bool createDirectory(const std::string& path);

// ============================================================================
// HÀM TIỆN ÍCH THỜI GIAN
// ============================================================================

/// @brief Lấy timestamp hiện tại dưới dạng chuỗi định dạng "YYYY-MM-DD HH:MM:SS"
/// Dùng cho logging, đặt tên file ảnh chụp, và hiển thị trên HUD
/// @return Chuỗi timestamp đã định dạng theo chuẩn ISO-like
std::string getCurrentTimestamp();

// ============================================================================
// NAMESPACE COLORS - HẰNG SỐ MÀU SẮC BGR CHO OpenCV
// ============================================================================

/// @brief Namespace chứa các hằng số màu sắc BGR (Blue-Green-Red) cho OpenCV
/// OpenCV sử dụng thứ tự BGR thay vì RGB thông thường - cần lưu ý thứ tự kênh
/// Các màu này được dùng thống nhất trong toàn bộ dự án để đảm bảo giao diện nhất quán
namespace Colors {
    const cv::Scalar GREEN(0, 255, 0);         // Màu xanh lá - dùng cho trạng thái tốt, FPS cao, phát hiện thành công
    const cv::Scalar RED(0, 0, 255);           // Màu đỏ - dùng cho cảnh báo, lỗi, pin yếu, FPS thấp
    const cv::Scalar BLUE(255, 0, 0);          // Màu xanh dương - dùng cho thông tin, viền đối tượng
    const cv::Scalar YELLOW(0, 255, 255);      // Màu vàng - dùng cho cảnh báo nhẹ, FPS trung bình
    const cv::Scalar CYAN(255, 255, 0);        // Màu xanh ngọc - dùng cho thông tin phụ, tọa độ GPS
    const cv::Scalar MAGENTA(255, 0, 255);     // Màu hồng tím - dùng cho nhãn đặc biệt, chế độ bay
    const cv::Scalar WHITE(255, 255, 255);     // Màu trắng - dùng cho text chính, vạch chia, viền HUD
    const cv::Scalar BLACK(0, 0, 0);           // Màu đen - dùng cho nền, viền text shadow
    const cv::Scalar ORANGE(0, 165, 255);      // Màu cam - dùng cho cảnh báo trung bình, pin trung bình
    const cv::Scalar DARK_GREEN(0, 128, 0);    // Màu xanh lá đậm - dùng cho phân biệt với xanh lá chính
} // Kết thúc namespace Colors

// ============================================================================
// HÀM LOGGING - GHI LOG RA CONSOLE CÓ MÀU VÀ TIMESTAMP
// ============================================================================

/// @brief Ghi log thông tin với tiền tố [INFO] và timestamp
/// Dùng cho các thông điệp thông thường như khởi tạo thành công, kết quả xử lý
/// @param msg Nội dung thông điệp cần ghi log
void logInfo(const std::string& msg);

/// @brief Ghi log cảnh báo với tiền tố [WARN] và timestamp
/// Dùng cho các tình huống bất thường nhưng chưa phải lỗi nghiêm trọng
/// @param msg Nội dung cảnh báo cần ghi log
void logWarning(const std::string& msg);

/// @brief Ghi log lỗi với tiền tố [ERROR] và timestamp
/// Dùng cho các lỗi nghiêm trọng cần xử lý ngay lập tức
/// @param msg Nội dung lỗi cần ghi log
void logError(const std::string& msg);

// ============================================================================
// HÀM VẼ TEXT VỚI NỀN - TIỆN ÍCH ĐỒ HỌA
// ============================================================================

/// @brief Vẽ text có nền hình chữ nhật đặc để text nổi bật trên mọi ảnh nền
/// Phương thức: đo kích thước text trước, vẽ hình chữ nhật nền, rồi vẽ text lên trên
/// @param frame Tham chiếu đến khung hình để vẽ lên
/// @param text Nội dung text cần hiển thị
/// @param pos Vị trí góc dưới trái của text (baseline) trên khung hình
/// @param font_face Kiểu font OpenCV - mặc định FONT_HERSHEY_SIMPLEX
/// @param font_scale Tỷ lệ phóng to font - mặc định 0.6
/// @param text_color Màu sắc BGR của text - mặc định trắng
/// @param bg_color Màu sắc BGR của nền hình chữ nhật - mặc định đen
/// @param thickness Độ dày nét vẽ text tính bằng pixel - mặc định 1
/// @param padding Khoảng đệm giữa text và viền hình chữ nhật nền - mặc định 5 pixel
void drawTextWithBackground(
    cv::Mat& frame,                                           // Khung hình đích để vẽ lên
    const std::string& text,                                  // Chuỗi text cần hiển thị
    cv::Point pos,                                            // Vị trí baseline của text
    int font_face = cv::FONT_HERSHEY_SIMPLEX,                // Font chữ OpenCV
    double font_scale = 0.6,                                  // Tỷ lệ kích thước font
    cv::Scalar text_color = cv::Scalar(255, 255, 255),        // Màu text (BGR) - mặc định trắng
    cv::Scalar bg_color = cv::Scalar(0, 0, 0),               // Màu nền (BGR) - mặc định đen
    int thickness = 1,                                        // Độ dày nét chữ
    int padding = 5                                           // Khoảng đệm xung quanh text
);

} // Kết thúc namespace drone_utils - đóng phạm vi namespace chính của module tiện ích
