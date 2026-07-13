#pragma once // Chỉ thị tiền xử lý đảm bảo file header chỉ được include một lần, tránh lỗi định nghĩa trùng lặp

// ============================================================================
// FILE: include/drone_hud.hpp
// PROJECT: DroneVisionPro - Hệ thống thị giác máy tính cho drone
// PURPOSE: Lớp DroneHUD - Hiển thị Head-Up Display overlay trên video drone
// AUTHOR: Tran Ngoc Bao - 24021238
// DATE: 2026-07-13
// DESCRIPTION: Cung cấp giao diện HUD chuyên nghiệp với la bàn, thanh độ cao,
//              thanh trạng thái, FPS, cảnh báo và các thông tin bay quan trọng
// ============================================================================

#include <opencv2/opencv.hpp>    // Thư viện OpenCV chính - cung cấp tất cả chức năng xử lý ảnh và vẽ đồ họa
#include <string>                // Thư viện chuỗi C++ chuẩn - dùng cho các thông điệp và nhãn hiển thị
#include <chrono>                // Thư viện thời gian C++ - dùng để lấy timestamp và tính toán thời gian nhấp nháy
#include <cmath>                 // Thư viện toán học - dùng cho sin, cos khi vẽ la bàn tròn
#include <iomanip>               // Thư viện định dạng I/O - dùng để format số thập phân và thời gian
#include <sstream>               // Thư viện string stream - dùng để tạo chuỗi định dạng phức tạp

/// @brief Cấu trúc chứa toàn bộ dữ liệu telemetry (đo từ xa) của drone
/// Struct này tập trung tất cả thông tin trạng thái bay để dễ quản lý và truyền dữ liệu
struct HUDData {
    double fps = 0.0;               // Số khung hình trên giây hiện tại - đo hiệu suất xử lý video
    double altitude = 0.0;           // Độ cao bay tính bằng mét - quan trọng cho an toàn bay
    double heading = 0.0;            // Hướng la bàn từ 0 đến 360 độ - 0/360 = Bắc, 90 = Đông, 180 = Nam, 270 = Tây
    int battery_percent = 100;       // Phần trăm pin còn lại - cảnh báo khi pin yếu để hạ cánh kịp thời
    double gps_latitude = 0.0;       // Vĩ độ GPS - tọa độ vị trí Bắc/Nam trên bản đồ
    double gps_longitude = 0.0;      // Kinh độ GPS - tọa độ vị trí Đông/Tây trên bản đồ
    std::string flight_mode = "STABILIZE"; // Chế độ bay hiện tại - STABILIZE/LOITER/AUTO/LAND/RTL
    int detection_count = 0;         // Số đối tượng được phát hiện - kết quả từ module nhận diện
    std::string warning_message = ""; // Thông điệp cảnh báo - hiển thị nhấp nháy màu đỏ khi có nguy hiểm
};

/// @brief Lớp DroneHUD - Vẽ giao diện Head-Up Display lên khung hình video drone
/// Lớp này cung cấp overlay HUD chuyên nghiệp giống như trong cockpit máy bay
/// bao gồm: la bàn, thanh độ cao, crosshair, thông tin pin, GPS, FPS, cảnh báo
class DroneHUD {
public:
    // ========================================================================
    // CONSTRUCTOR VÀ DESTRUCTOR
    // ========================================================================

    /// @brief Constructor khởi tạo HUD với kích thước khung hình cụ thể
    /// @param frame_width Chiều rộng khung hình tính bằng pixel - dùng để tính vị trí các phần tử HUD
    /// @param frame_height Chiều cao khung hình tính bằng pixel - dùng để tính vị trí các phần tử HUD
    DroneHUD(int frame_width, int frame_height);

    /// @brief Destructor mặc định - giải phóng tài nguyên khi đối tượng bị hủy
    ~DroneHUD() = default;

    // ========================================================================
    // PHƯƠNG THỨC VẼ HUD CHÍNH
    // ========================================================================

    /// @brief Vẽ toàn bộ overlay HUD lên khung hình video
    /// Phương thức này gọi lần lượt tất cả các phương thức vẽ thành phần con
    /// @param frame Tham chiếu đến khung hình OpenCV - sẽ được vẽ trực tiếp lên đây
    void drawHUD(cv::Mat& frame);

    // ========================================================================
    // CÁC PHƯƠNG THỨC CẬP NHẬT DỮ LIỆU TELEMETRY
    // ========================================================================

    /// @brief Cập nhật giá trị FPS hiện tại để hiển thị trên HUD
    /// @param fps Số khung hình trên giây - giá trị thực dương
    void setFPS(double fps);

    /// @brief Cập nhật độ cao bay hiện tại
    /// @param altitude Độ cao tính bằng mét so với mặt đất - giá trị thực không âm
    void setAltitude(double altitude);

    /// @brief Cập nhật hướng la bàn hiện tại của drone
    /// @param heading Góc hướng từ 0 đến 360 độ - 0=Bắc, 90=Đông, 180=Nam, 270=Tây
    void setHeading(double heading);

    /// @brief Cập nhật mức pin còn lại
    /// @param percent Phần trăm pin từ 0 đến 100 - cảnh báo khi dưới 20%
    void setBattery(int percent);

    /// @brief Cập nhật tọa độ GPS hiện tại của drone
    /// @param lat Vĩ độ GPS - giá trị từ -90 đến +90 (Bắc dương, Nam âm)
    /// @param lon Kinh độ GPS - giá trị từ -180 đến +180 (Đông dương, Tây âm)
    void setGPSCoords(double lat, double lon);

    /// @brief Cập nhật chế độ bay hiện tại
    /// @param mode Chuỗi chế độ bay: "STABILIZE", "LOITER", "AUTO", "LAND", "RTL", "GUIDED"
    void setFlightMode(const std::string& mode);

    /// @brief Cập nhật số đối tượng được phát hiện bởi module nhận diện
    /// @param count Số lượng đối tượng - giá trị nguyên không âm
    void setDetectionCount(int count);

    /// @brief Đặt thông điệp cảnh báo hiển thị nhấp nháy trên HUD
    /// @param msg Nội dung cảnh báo - chuỗi rỗng để xóa cảnh báo
    void setWarning(const std::string& msg);

    // ========================================================================
    // CÁC PHƯƠNG THỨC BẬT/TẮT THÀNH PHẦN HUD
    // ========================================================================

    /// @brief Bật hoặc tắt hiển thị dấu chữ thập ngắm ở giữa màn hình
    /// @param enable true = hiển thị, false = ẩn crosshair
    void enableCrosshair(bool enable);

    /// @brief Bật hoặc tắt hiển thị la bàn tròn phía trên khung hình
    /// @param enable true = hiển thị, false = ẩn la bàn
    void enableCompass(bool enable);

    /// @brief Bật hoặc tắt hiển thị thanh độ cao dọc bên phải khung hình
    /// @param enable true = hiển thị, false = ẩn thanh độ cao
    void enableAltitudeBar(bool enable);

private:
    // ========================================================================
    // CÁC PHƯƠNG THỨC VẼ THÀNH PHẦN CON (PRIVATE)
    // ========================================================================

    /// @brief Vẽ dấu chữ thập ngắm ở giữa khung hình với nền bán trong suốt
    /// Crosshair giúp phi công xác định chính xác tâm camera khi ngắm mục tiêu
    /// @param frame Tham chiếu đến khung hình để vẽ lên
    void drawCrosshair(cv::Mat& frame);

    /// @brief Vẽ la bàn tròn với các vạch độ, nhãn N/S/E/W và kim chỉ hướng
    /// La bàn xoay theo hướng bay thực tế, giúp phi công biết drone đang quay về hướng nào
    /// @param frame Tham chiếu đến khung hình để vẽ lên
    void drawCompass(cv::Mat& frame);

    /// @brief Vẽ thanh độ cao dọc bên phải với vạch chia và con trỏ hiện tại
    /// Thanh độ cao hiển thị trực quan mức cao của drone so với mặt đất
    /// @param frame Tham chiếu đến khung hình để vẽ lên
    void drawAltitudeBar(cv::Mat& frame);

    /// @brief Vẽ thanh trạng thái ngang ở dưới cùng với thông tin pin, GPS, chế độ bay
    /// Thanh trạng thái tổng hợp các thông tin quan trọng nhất ở một vị trí dễ nhìn
    /// @param frame Tham chiếu đến khung hình để vẽ lên
    void drawStatusBar(cv::Mat& frame);

    /// @brief Vẽ bộ đếm FPS ở góc trên trái với màu thay đổi theo mức hiệu suất
    /// Xanh lá = tốt (>=30fps), Vàng = trung bình (>=15fps), Đỏ = kém (<15fps)
    /// @param frame Tham chiếu đến khung hình để vẽ lên
    void drawFPSCounter(cv::Mat& frame);

    /// @brief Vẽ nhãn thời gian hiện tại ở góc trên phải theo định dạng HH:MM:SS
    /// Timestamp giúp ghi nhận thời điểm chính xác trong video bay
    /// @param frame Tham chiếu đến khung hình để vẽ lên
    void drawTimestamp(cv::Mat& frame);

    /// @brief Vẽ thông điệp cảnh báo nhấp nháy màu đỏ ở giữa phía trên khung hình
    /// Hiệu ứng nhấp nháy tạo sự chú ý ngay lập tức cho phi công khi có nguy hiểm
    /// @param frame Tham chiếu đến khung hình để vẽ lên
    void drawWarning(cv::Mat& frame);

    /// @brief Vẽ tóm tắt số lượng đối tượng được phát hiện ở góc trên trái (dưới FPS)
    /// Hiển thị kết quả từ module nhận diện đối tượng
    /// @param frame Tham chiếu đến khung hình để vẽ lên
    void drawDetectionSummary(cv::Mat& frame);

    // ========================================================================
    // BIẾN THÀNH VIÊN
    // ========================================================================

    int frame_width_;       // Chiều rộng khung hình (pixel) - dùng để tính toán vị trí tương đối của các phần tử HUD
    int frame_height_;      // Chiều cao khung hình (pixel) - dùng để tính toán vị trí tương đối của các phần tử HUD

    HUDData hud_data_;      // Cấu trúc chứa toàn bộ dữ liệu telemetry hiện tại - được cập nhật bởi các setter

    bool show_crosshair_;   // Cờ điều khiển hiển thị crosshair - mặc định bật
    bool show_compass_;     // Cờ điều khiển hiển thị la bàn - mặc định bật
    bool show_altitude_bar_; // Cờ điều khiển hiển thị thanh độ cao - mặc định bật

    std::chrono::steady_clock::time_point last_warning_toggle_; // Thời điểm chuyển đổi trạng thái nhấp nháy cảnh báo
    bool warning_visible_;  // Trạng thái hiện/ẩn hiện tại của cảnh báo nhấp nháy - luân phiên true/false
};
