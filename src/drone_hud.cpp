// ============================================================================
// FILE: src/drone_hud.cpp
// PROJECT: DroneVisionPro - Hệ thống thị giác máy tính cho drone
// PURPOSE: Triển khai đầy đủ lớp DroneHUD - vẽ giao diện HUD lên video drone
// AUTHOR: Tran Ngoc Bao - 24021238
// DATE: 2026-07-13
// DESCRIPTION: File này chứa toàn bộ code vẽ các thành phần HUD bao gồm:
//              crosshair, la bàn, thanh độ cao, thanh trạng thái, FPS,
//              timestamp, cảnh báo nhấp nháy và tóm tắt nhận diện
// ============================================================================

#include "drone_hud.hpp" // Include header của lớp DroneHUD - chứa khai báo lớp và struct HUDData
#include "utils.hpp"     // Include header tiện ích - dùng hàm drawTextWithBackground và hằng số màu

#ifndef M_PI                       // Kiểm tra nếu hằng số PI chưa được định nghĩa (một số compiler không có sẵn)
#define M_PI 3.14159265358979323846 // Định nghĩa PI với độ chính xác cao - dùng cho tính toán góc la bàn
#endif                             // Kết thúc khối điều kiện tiền xử lý cho M_PI

// ============================================================================
// CONSTRUCTOR - KHỞI TẠO HUD VỚI KÍCH THƯỚC KHUNG HÌNH
// ============================================================================

/// @brief Constructor khởi tạo tất cả biến thành viên với giá trị mặc định
/// Tất cả thành phần HUD được bật mặc định, cảnh báo bắt đầu ở trạng thái hiện
DroneHUD::DroneHUD(int frame_width, int frame_height)
    : frame_width_(frame_width),           // Lưu chiều rộng khung hình để tính vị trí phần tử HUD
      frame_height_(frame_height),         // Lưu chiều cao khung hình để tính vị trí phần tử HUD
      show_crosshair_(true),               // Bật crosshair mặc định - phi công cần dấu ngắm ngay từ đầu
      show_compass_(true),                 // Bật la bàn mặc định - hướng bay là thông tin thiết yếu
      show_altitude_bar_(true),            // Bật thanh độ cao mặc định - an toàn bay cần biết độ cao
      warning_visible_(true)               // Cảnh báo bắt đầu ở trạng thái hiện - sẽ nhấp nháy khi có cảnh báo
{
    last_warning_toggle_ = std::chrono::steady_clock::now(); // Ghi nhận thời điểm bắt đầu để tính chu kỳ nhấp nháy
}

// ============================================================================
// PHƯƠNG THỨC VẼ HUD CHÍNH - TỔNG HỢP TẤT CẢ THÀNH PHẦN
// ============================================================================

/// @brief Vẽ toàn bộ HUD overlay lên khung hình theo thứ tự layer hợp lý
/// Thứ tự vẽ quan trọng: vẽ nền bán trong suốt trước, rồi vẽ text và đồ họa lên trên
void DroneHUD::drawHUD(cv::Mat& frame) {
    // Kiểm tra khung hình đầu vào có hợp lệ không - tránh crash khi frame rỗng
    if (frame.empty()) {       // empty() trả về true nếu Ma trận không chứa dữ liệu pixel nào
        return;                // Thoát ngay nếu frame rỗng - không có gì để vẽ lên
    }

    // Cập nhật kích thước khung hình trong trường hợp frame thay đổi kích thước giữa chừng
    frame_width_ = frame.cols;   // cols là số cột = chiều rộng của Ma trận ảnh
    frame_height_ = frame.rows;  // rows là số hàng = chiều cao của Ma trận ảnh

    // Vẽ từng thành phần HUD theo thứ tự: nền -> chính -> overlay
    // Thứ tự này đảm bảo các thành phần quan trọng hơn nằm trên cùng

    drawFPSCounter(frame);        // Vẽ FPS ở góc trên trái - ưu tiên vẽ sớm vì nằm ở góc
    drawTimestamp(frame);         // Vẽ timestamp ở góc trên phải - thông tin thời gian cơ bản

    // Vẽ các thành phần HUD có điều kiện bật/tắt
    if (show_crosshair_) {        // Kiểm tra cờ bật/tắt crosshair trước khi vẽ
        drawCrosshair(frame);     // Vẽ dấu chữ thập ngắm ở giữa màn hình
    }

    if (show_compass_) {          // Kiểm tra cờ bật/tắt la bàn trước khi vẽ
        drawCompass(frame);       // Vẽ la bàn tròn phía trên giữa khung hình
    }

    if (show_altitude_bar_) {     // Kiểm tra cờ bật/tắt thanh độ cao trước khi vẽ
        drawAltitudeBar(frame);   // Vẽ thanh độ cao dọc bên phải khung hình
    }

    drawStatusBar(frame);         // Vẽ thanh trạng thái ngang ở dưới cùng - luôn hiển thị
    drawDetectionSummary(frame);  // Vẽ tóm tắt nhận diện - luôn hiển thị nếu có detection
    drawWarning(frame);           // Vẽ cảnh báo cuối cùng để nằm trên tất cả - ưu tiên cao nhất
}

// ============================================================================
// CÁC PHƯƠNG THỨC SETTER - CẬP NHẬT DỮ LIỆU TELEMETRY
// ============================================================================

/// @brief Cập nhật FPS - gán trực tiếp vào struct HUDData
void DroneHUD::setFPS(double fps) {
    hud_data_.fps = fps; // Lưu giá trị FPS mới - sẽ được dùng khi vẽ FPS counter
}

/// @brief Cập nhật độ cao - gán trực tiếp vào struct HUDData
void DroneHUD::setAltitude(double altitude) {
    hud_data_.altitude = altitude; // Lưu độ cao mới (mét) - sẽ được dùng khi vẽ thanh độ cao
}

/// @brief Cập nhật hướng la bàn - chuẩn hóa về khoảng [0, 360)
void DroneHUD::setHeading(double heading) {
    // Dùng fmod để chuẩn hóa góc về khoảng [0, 360) - xử lý cả giá trị âm và lớn hơn 360
    hud_data_.heading = std::fmod(heading, 360.0); // fmod tính phần dư phép chia cho số thực
    if (hud_data_.heading < 0) {                   // Trường hợp heading âm (ví dụ: -30 độ)
        hud_data_.heading += 360.0;                // Chuyển về giá trị dương tương đương (330 độ)
    }
}

/// @brief Cập nhật phần trăm pin - giới hạn trong khoảng [0, 100]
void DroneHUD::setBattery(int percent) {
    // std::clamp giới hạn giá trị trong phạm vi an toàn - tránh hiển thị giá trị vô lý
    hud_data_.battery_percent = std::clamp(percent, 0, 100); // Đảm bảo pin luôn từ 0% đến 100%
}

/// @brief Cập nhật tọa độ GPS - lưu cả vĩ độ và kinh độ cùng lúc
void DroneHUD::setGPSCoords(double lat, double lon) {
    hud_data_.gps_latitude = lat;   // Lưu vĩ độ GPS - giá trị từ -90 (cực Nam) đến +90 (cực Bắc)
    hud_data_.gps_longitude = lon;  // Lưu kinh độ GPS - giá trị từ -180 (Tây) đến +180 (Đông)
}

/// @brief Cập nhật chế độ bay - copy chuỗi mode vào struct
void DroneHUD::setFlightMode(const std::string& mode) {
    hud_data_.flight_mode = mode; // Lưu chuỗi chế độ bay - ví dụ: "STABILIZE", "LOITER", "AUTO"
}

/// @brief Cập nhật số lượng đối tượng phát hiện được
void DroneHUD::setDetectionCount(int count) {
    hud_data_.detection_count = std::max(0, count); // Đảm bảo số lượng không âm - max với 0
}

/// @brief Đặt thông điệp cảnh báo - chuỗi rỗng để xóa cảnh báo
void DroneHUD::setWarning(const std::string& msg) {
    hud_data_.warning_message = msg; // Lưu thông điệp cảnh báo - sẽ hiển thị nhấp nháy nếu không rỗng
}

// ============================================================================
// CÁC PHƯƠNG THỨC BẬT/TẮT THÀNH PHẦN HUD
// ============================================================================

/// @brief Bật/tắt crosshair - dấu ngắm ở giữa màn hình
void DroneHUD::enableCrosshair(bool enable) {
    show_crosshair_ = enable; // true = hiển thị crosshair, false = ẩn crosshair
}

/// @brief Bật/tắt la bàn - vòng tròn la bàn với N/S/E/W
void DroneHUD::enableCompass(bool enable) {
    show_compass_ = enable; // true = hiển thị la bàn, false = ẩn la bàn
}

/// @brief Bật/tắt thanh độ cao - thanh dọc bên phải
void DroneHUD::enableAltitudeBar(bool enable) {
    show_altitude_bar_ = enable; // true = hiển thị thanh độ cao, false = ẩn thanh độ cao
}

// ============================================================================
// VẼ CROSSHAIR - DẤU CHỮ THẬP NGẮM Ở GIỮA MÀN HÌNH
// ============================================================================

/// @brief Vẽ dấu chữ thập ngắm với nền bán trong suốt ở chính giữa khung hình
/// Crosshair gồm: vòng tròn bán trong suốt + 4 đường ngang dọc + dấu chấm tâm
void DroneHUD::drawCrosshair(cv::Mat& frame) {
    // Tính tọa độ tâm khung hình - đây là vị trí đặt crosshair
    int center_x = frame_width_ / 2;    // Tọa độ X tâm = nửa chiều rộng
    int center_y = frame_height_ / 2;   // Tọa độ Y tâm = nửa chiều cao

    int cross_size = 30;  // Chiều dài nửa cánh chữ thập tính bằng pixel - đủ lớn để dễ nhìn
    int gap = 8;          // Khoảng trống ở giữa crosshair - tạo khoảng hở quanh tâm để thấy mục tiêu

    // === Bước 1: Vẽ nền bán trong suốt hình tròn quanh crosshair ===
    // Kỹ thuật: tạo overlay riêng, vẽ lên đó, rồi trộn với frame gốc bằng addWeighted

    cv::Mat overlay = frame.clone(); // Tạo bản sao hoàn toàn của frame - dùng làm lớp overlay vẽ tạm

    // Vẽ hình tròn đen nền với bán kính lớn hơn crosshair một chút
    cv::circle(overlay,                          // Vẽ lên overlay (không phải frame gốc)
               cv::Point(center_x, center_y),    // Tâm hình tròn tại tâm frame
               cross_size + 10,                  // Bán kính = kích thước crosshair + 10 pixel đệm
               cv::Scalar(0, 0, 0),              // Màu đen cho nền
               cv::FILLED);                      // FILLED = tô đặc hình tròn (không chỉ viền)

    // Trộn overlay bán trong suốt với frame gốc sử dụng alpha blending
    // Công thức: result = alpha * overlay + (1-alpha) * frame
    double alpha = 0.3;                          // Hệ số alpha = 0.3 = 30% overlay, 70% frame gốc
    cv::addWeighted(overlay, alpha,              // Overlay với trọng số alpha (30% overlay)
                    frame, 1.0 - alpha,          // Frame gốc với trọng số 1-alpha (70% gốc)
                    0,                           // Gamma = 0 (không thêm hằng số sáng)
                    frame);                      // Kết quả ghi đè lên frame gốc

    // === Bước 2: Vẽ 4 nhánh chữ thập với khoảng trống ở giữa ===

    cv::Scalar cross_color(0, 255, 0); // Màu xanh lá cho crosshair - màu chuẩn HUD quân sự
    int line_thickness = 2;            // Độ dày đường kẻ = 2 pixel - rõ ràng mà không quá đậm

    // Vẽ nhánh trái: từ (center - cross_size) đến (center - gap) theo trục X
    cv::line(frame,                                             // Vẽ trực tiếp lên frame
             cv::Point(center_x - cross_size, center_y),        // Điểm bắt đầu bên trái
             cv::Point(center_x - gap, center_y),               // Điểm kết thúc cách tâm một khoảng gap
             cross_color,                                       // Màu xanh lá
             line_thickness);                                    // Độ dày 2 pixel

    // Vẽ nhánh phải: từ (center + gap) đến (center + cross_size) theo trục X
    cv::line(frame,                                             // Vẽ trực tiếp lên frame
             cv::Point(center_x + gap, center_y),               // Điểm bắt đầu cách tâm một khoảng gap
             cv::Point(center_x + cross_size, center_y),        // Điểm kết thúc bên phải
             cross_color,                                       // Màu xanh lá
             line_thickness);                                    // Độ dày 2 pixel

    // Vẽ nhánh trên: từ (center - cross_size) đến (center - gap) theo trục Y
    cv::line(frame,                                             // Vẽ trực tiếp lên frame
             cv::Point(center_x, center_y - cross_size),        // Điểm bắt đầu phía trên
             cv::Point(center_x, center_y - gap),               // Điểm kết thúc cách tâm một khoảng gap
             cross_color,                                       // Màu xanh lá
             line_thickness);                                    // Độ dày 2 pixel

    // Vẽ nhánh dưới: từ (center + gap) đến (center + cross_size) theo trục Y
    cv::line(frame,                                             // Vẽ trực tiếp lên frame
             cv::Point(center_x, center_y + gap),               // Điểm bắt đầu cách tâm một khoảng gap
             cv::Point(center_x, center_y + cross_size),        // Điểm kết thúc phía dưới
             cross_color,                                       // Màu xanh lá
             line_thickness);                                    // Độ dày 2 pixel

    // === Bước 3: Vẽ dấu chấm nhỏ ở chính tâm crosshair ===
    cv::circle(frame,                            // Vẽ trực tiếp lên frame
               cv::Point(center_x, center_y),    // Vị trí tâm chính xác
               2,                                // Bán kính 2 pixel - nhỏ gọn
               cross_color,                      // Màu xanh lá đồng nhất với crosshair
               cv::FILLED);                      // Tô đặc dấu chấm

    // === Bước 4: Vẽ vòng tròn ngoài cùng của crosshair ===
    cv::circle(frame,                            // Vẽ trực tiếp lên frame
               cv::Point(center_x, center_y),    // Tâm vòng tròn tại tâm frame
               cross_size + 5,                   // Bán kính lớn hơn nhánh chữ thập 5 pixel
               cross_color,                      // Màu xanh lá
               1);                               // Độ dày viền = 1 pixel - mỏng thanh lịch
}

// ============================================================================
// VẼ LA BÀN TRÒN - COMPASS VỚI VẠCH ĐỘ VÀ KIM CHỈ HƯỚNG
// ============================================================================

/// @brief Vẽ la bàn tròn với vạch chia độ, nhãn N/S/E/W và kim chỉ hướng xoay
/// La bàn hiển thị ở phía trên giữa khung hình, xoay theo heading thực tế của drone
void DroneHUD::drawCompass(cv::Mat& frame) {
    // Tính vị trí tâm la bàn - đặt ở trên giữa khung hình
    int compass_x = frame_width_ / 2;   // Tọa độ X = giữa khung hình theo chiều ngang
    int compass_y = 70;                  // Tọa độ Y = cách mép trên 70 pixel - vừa đủ không che FPS
    int radius = 50;                     // Bán kính la bàn = 50 pixel - kích thước vừa phải

    // === Bước 1: Vẽ nền bán trong suốt cho la bàn ===
    cv::Mat overlay = frame.clone();     // Clone frame để tạo overlay bán trong suốt
    cv::circle(overlay,                  // Vẽ hình tròn nền lên overlay
               cv::Point(compass_x, compass_y), // Tâm la bàn
               radius + 15,             // Bán kính lớn hơn la bàn 15 pixel để có padding
               cv::Scalar(0, 0, 0),     // Màu đen cho nền
               cv::FILLED);             // Tô đặc hình tròn nền

    // Trộn overlay vào frame với alpha = 0.4 (40% đen, 60% ảnh gốc)
    cv::addWeighted(overlay, 0.4,        // 40% overlay đen bán trong suốt
                    frame, 0.6,          // 60% frame gốc giữ lại
                    0, frame);           // Kết quả ghi vào frame

    // === Bước 2: Vẽ vòng tròn chính của la bàn ===
    cv::circle(frame,                              // Vẽ lên frame (đã trộn overlay)
               cv::Point(compass_x, compass_y),    // Tâm la bàn
               radius,                             // Bán kính chính xác
               cv::Scalar(0, 255, 0),              // Màu xanh lá - phong cách HUD
               2);                                 // Độ dày viền = 2 pixel

    // === Bước 3: Vẽ các vạch chia độ xung quanh la bàn ===
    // Vẽ 36 vạch (mỗi 10 độ), vạch chính dài hơn tại mỗi 30 độ
    for (int deg = 0; deg < 360; deg += 10) { // Lặp từ 0 đến 350 độ, bước 10 độ
        // Tính góc thực tế sau khi xoay theo heading của drone
        // Trừ heading để la bàn xoay ngược - khi drone quay phải, la bàn quay trái
        double angle_rad = (deg - hud_data_.heading - 90) * M_PI / 180.0; // Chuyển sang radian, -90 để 0° = Bắc (trên)

        // Xác định chiều dài vạch: vạch chính (mỗi 30°) dài hơn vạch phụ (mỗi 10°)
        int tick_inner, tick_outer;        // Bán kính trong và ngoài của vạch
        if (deg % 30 == 0) {              // Vạch chính tại 0, 30, 60, 90, ... độ
            tick_inner = radius - 12;      // Bắt đầu cách tâm = radius - 12 pixel
            tick_outer = radius;           // Kết thúc tại viền la bàn
        } else {                          // Vạch phụ tại 10, 20, 40, 50, ... độ
            tick_inner = radius - 6;       // Vạch ngắn hơn - bắt đầu cách viền 6 pixel
            tick_outer = radius;           // Kết thúc tại viền la bàn
        }

        // Tính tọa độ 2 đầu vạch bằng công thức lượng giác: x = r*cos(θ), y = r*sin(θ)
        cv::Point p1(compass_x + static_cast<int>(tick_inner * std::cos(angle_rad)),  // Điểm trong của vạch
                     compass_y + static_cast<int>(tick_inner * std::sin(angle_rad))); // cos/sin tính từ góc radian
        cv::Point p2(compass_x + static_cast<int>(tick_outer * std::cos(angle_rad)),  // Điểm ngoài của vạch
                     compass_y + static_cast<int>(tick_outer * std::sin(angle_rad))); // Nằm trên viền la bàn

        // Vạch chính (30°) màu trắng sáng, vạch phụ (10°) màu xám nhạt
        cv::Scalar tick_color = (deg % 30 == 0) ? cv::Scalar(255, 255, 255)  // Trắng cho vạch chính
                                                : cv::Scalar(150, 150, 150); // Xám cho vạch phụ

        cv::line(frame, p1, p2, tick_color, 1); // Vẽ đường thẳng nối 2 đầu vạch với độ dày 1 pixel
    }

    // === Bước 4: Vẽ nhãn N/S/E/W tại 4 hướng chính ===
    // Mảng chứa 4 hướng chính với ký hiệu và góc tương ứng
    struct CompassLabel {         // Cấu trúc lưu thông tin nhãn hướng
        const char* label;       // Ký hiệu hướng: "N", "S", "E", "W"
        int degree;              // Góc tương ứng: N=0, E=90, S=180, W=270
        cv::Scalar color;        // Màu sắc riêng cho từng hướng
    };

    // Định nghĩa 4 hướng chính với màu sắc phân biệt
    CompassLabel labels[] = {
        {"N", 0,   cv::Scalar(0, 0, 255)},     // Bắc = màu ĐỎ - hướng chính quan trọng nhất
        {"E", 90,  cv::Scalar(255, 255, 255)},  // Đông = màu TRẮNG
        {"S", 180, cv::Scalar(255, 255, 255)},  // Nam = màu TRẮNG
        {"W", 270, cv::Scalar(255, 255, 255)}   // Tây = màu TRẮNG
    };

    // Vẽ từng nhãn hướng tại vị trí tương ứng trên la bàn
    for (const auto& lbl : labels) { // Duyệt qua mảng 4 nhãn hướng
        // Tính góc radian cho vị trí nhãn - xoay theo heading giống vạch chia
        double angle_rad = (lbl.degree - hud_data_.heading - 90) * M_PI / 180.0; // Công thức xoay la bàn
        int label_radius = radius - 22; // Đặt nhãn bên trong vạch chia - cách tâm = radius - 22

        // Tính tọa độ đặt nhãn bằng lượng giác
        int label_x = compass_x + static_cast<int>(label_radius * std::cos(angle_rad)) - 5; // -5 để canh giữa ký tự
        int label_y = compass_y + static_cast<int>(label_radius * std::sin(angle_rad)) + 5; // +5 để canh giữa theo chiều dọc

        // Vẽ nhãn hướng lên frame
        cv::putText(frame,                              // Vẽ lên frame
                    lbl.label,                          // Ký hiệu hướng: "N", "S", "E", "W"
                    cv::Point(label_x, label_y),        // Vị trí đặt text
                    cv::FONT_HERSHEY_SIMPLEX,           // Font đơn giản, dễ đọc
                    0.5,                                // Kích thước font = 0.5 - vừa phải
                    lbl.color,                          // Màu sắc riêng (đỏ cho Bắc, trắng cho còn lại)
                    2);                                 // Độ dày = 2 pixel cho rõ ràng
    }

    // === Bước 5: Vẽ kim chỉ hướng (tam giác) phía trên la bàn ===
    // Kim cố định ở vị trí 12 giờ (trên cùng) - la bàn xoay bên dưới

    // Tạo 3 đỉnh của tam giác kim chỉ hướng
    std::vector<cv::Point> arrow_pts; // Vector chứa các đỉnh tam giác
    arrow_pts.push_back(cv::Point(compass_x, compass_y - radius - 10));     // Đỉnh trên (mũi nhọn) - nhô ra ngoài vòng la bàn
    arrow_pts.push_back(cv::Point(compass_x - 6, compass_y - radius + 2));  // Đỉnh trái dưới
    arrow_pts.push_back(cv::Point(compass_x + 6, compass_y - radius + 2));  // Đỉnh phải dưới

    // Vẽ tam giác tô đặc làm kim chỉ hướng
    cv::fillConvexPoly(frame,         // Vẽ trực tiếp lên frame
                       arrow_pts,     // Các đỉnh tam giác
                       cv::Scalar(0, 0, 255)); // Màu đỏ - nổi bật trên nền xanh lá

    // === Bước 6: Hiển thị giá trị heading bằng số ở dưới la bàn ===
    std::ostringstream heading_text;                    // Tạo string stream để format chuỗi
    heading_text << std::fixed << std::setprecision(0); // Không hiện số thập phân - heading lấy số nguyên
    heading_text << hud_data_.heading << "\xC2\xB0";    // Thêm ký hiệu độ (°) dạng UTF-8

    // Đo kích thước text để canh giữa
    int baseline = 0;                       // Biến nhận baseline - khoảng cách từ dòng cơ sở đến đáy ký tự
    cv::Size text_size = cv::getTextSize(   // Đo kích thước pixel của chuỗi text
        heading_text.str(),                 // Chuỗi cần đo
        cv::FONT_HERSHEY_SIMPLEX,           // Font chữ
        0.5,                                // Kích thước font
        1,                                  // Độ dày nét
        &baseline);                         // Nhận giá trị baseline

    // Vẽ text heading canh giữa dưới la bàn
    cv::putText(frame,                                                      // Vẽ lên frame
                heading_text.str(),                                         // Chuỗi heading (ví dụ: "045°")
                cv::Point(compass_x - text_size.width / 2,                  // X canh giữa: tâm - nửa chiều rộng text
                          compass_y + radius + 20),                         // Y dưới la bàn 20 pixel
                cv::FONT_HERSHEY_SIMPLEX,                                   // Font đơn giản
                0.5,                                                        // Kích thước font
                cv::Scalar(0, 255, 0),                                      // Màu xanh lá
                1);                                                         // Độ dày 1 pixel
}

// ============================================================================
// VẼ THANH ĐỘ CAO - ALTITUDE BAR DỌC BÊN PHẢI
// ============================================================================

/// @brief Vẽ thanh độ cao dọc bên phải khung hình với vạch chia và con trỏ
/// Thanh hiển thị phạm vi 0-100m mặc định, con trỏ chỉ độ cao hiện tại
void DroneHUD::drawAltitudeBar(cv::Mat& frame) {
    // Định nghĩa kích thước và vị trí thanh độ cao
    int bar_x = frame_width_ - 60;    // Tọa độ X = cách mép phải 60 pixel
    int bar_top = 80;                 // Tọa độ Y đỉnh thanh = cách mép trên 80 pixel (tránh timestamp)
    int bar_bottom = frame_height_ - 80; // Tọa độ Y đáy thanh = cách mép dưới 80 pixel (tránh status bar)
    int bar_width = 20;               // Chiều rộng thanh = 20 pixel - đủ để thấy nhưng không che ảnh
    int bar_height = bar_bottom - bar_top; // Chiều cao thanh = khoảng cách đỉnh đến đáy

    // === Bước 1: Vẽ nền bán trong suốt cho khu vực thanh độ cao ===
    cv::Mat overlay = frame.clone();  // Clone frame cho overlay bán trong suốt

    // Vẽ hình chữ nhật nền bao quanh thanh độ cao
    cv::rectangle(overlay,                                           // Vẽ lên overlay
                  cv::Point(bar_x - 30, bar_top - 20),              // Góc trên trái (mở rộng thêm cho nhãn)
                  cv::Point(bar_x + bar_width + 15, bar_bottom + 20), // Góc dưới phải (mở rộng thêm)
                  cv::Scalar(0, 0, 0),                               // Màu đen
                  cv::FILLED);                                       // Tô đặc

    cv::addWeighted(overlay, 0.4,    // 40% overlay đen
                    frame, 0.6,      // 60% ảnh gốc
                    0, frame);       // Ghi kết quả vào frame

    // === Bước 2: Vẽ khung thanh độ cao ===
    cv::rectangle(frame,                                    // Vẽ lên frame đã blend
                  cv::Point(bar_x, bar_top),                // Góc trên trái thanh
                  cv::Point(bar_x + bar_width, bar_bottom), // Góc dưới phải thanh
                  cv::Scalar(0, 255, 0),                    // Viền xanh lá - phong cách HUD
                  1);                                       // Độ dày viền = 1 pixel

    // === Bước 3: Xác định phạm vi hiển thị và tính toán tỷ lệ ===
    double max_altitude = 100.0;  // Độ cao tối đa hiển thị = 100 mét - đủ cho hầu hết drone dân dụng

    // Giới hạn altitude hiển thị trong phạm vi hợp lệ
    double display_alt = std::clamp(hud_data_.altitude, 0.0, max_altitude); // Clamp về [0, 100]

    // Tính vị trí Y của con trỏ trên thanh - đáy = 0m, đỉnh = max_altitude
    // Công thức: Y giảm khi altitude tăng (vì Y tăng xuống dưới trong OpenCV)
    double ratio = display_alt / max_altitude;                      // Tỷ lệ altitude so với max (0.0 đến 1.0)
    int pointer_y = bar_bottom - static_cast<int>(ratio * bar_height); // Chuyển tỷ lệ thành vị trí pixel Y

    // === Bước 4: Tô màu phần thanh từ đáy đến vị trí altitude hiện tại ===
    if (pointer_y < bar_bottom) {  // Chỉ tô nếu altitude > 0 (con trỏ không ở đáy)
        // Tạo gradient màu: xanh lá (thấp) -> vàng (trung bình) -> đỏ (cao)
        cv::Scalar fill_color;                             // Biến lưu màu tô
        if (display_alt < 30.0) {                          // Dưới 30m: xanh lá = an toàn
            fill_color = cv::Scalar(0, 200, 0);            // Xanh lá đậm vừa
        } else if (display_alt < 70.0) {                   // 30-70m: vàng = cẩn thận
            fill_color = cv::Scalar(0, 200, 200);          // Vàng đậm vừa
        } else {                                           // Trên 70m: đỏ = nguy hiểm
            fill_color = cv::Scalar(0, 0, 200);            // Đỏ đậm vừa
        }

        // Tô hình chữ nhật từ vị trí con trỏ đến đáy thanh
        cv::rectangle(frame,                                        // Vẽ lên frame
                      cv::Point(bar_x + 1, pointer_y),              // Góc trên = vị trí con trỏ (+1 tránh đè viền)
                      cv::Point(bar_x + bar_width - 1, bar_bottom - 1), // Góc dưới = đáy thanh (-1 tránh đè viền)
                      fill_color,                                   // Màu tô theo mức altitude
                      cv::FILLED);                                  // Tô đặc hình chữ nhật
    }

    // === Bước 5: Vẽ các vạch chia trên thanh (mỗi 10m) ===
    for (int alt = 0; alt <= static_cast<int>(max_altitude); alt += 10) { // Lặp mỗi 10 mét
        // Tính vị trí Y cho vạch chia này
        double alt_ratio = static_cast<double>(alt) / max_altitude; // Tỷ lệ altitude cho vạch
        int tick_y = bar_bottom - static_cast<int>(alt_ratio * bar_height); // Chuyển sang pixel Y

        // Vẽ vạch ngang nhỏ ở bên trái thanh
        cv::line(frame,                                     // Vẽ lên frame
                 cv::Point(bar_x - 5, tick_y),              // Bắt đầu bên trái thanh 5 pixel
                 cv::Point(bar_x, tick_y),                   // Kết thúc tại viền trái thanh
                 cv::Scalar(255, 255, 255),                  // Màu trắng cho vạch chia
                 1);                                         // Độ dày 1 pixel

        // Vẽ nhãn số (giá trị altitude) bên trái vạch
        std::string alt_label = std::to_string(alt);        // Chuyển số nguyên thành chuỗi
        cv::putText(frame,                                   // Vẽ lên frame
                    alt_label,                               // Nhãn số: "0", "10", "20", ...
                    cv::Point(bar_x - 30, tick_y + 4),       // Vị trí bên trái vạch, +4 để canh giữa dọc
                    cv::FONT_HERSHEY_PLAIN,                  // Font đơn giản nhỏ gọn
                    0.8,                                     // Kích thước font nhỏ
                    cv::Scalar(200, 200, 200),               // Màu xám sáng
                    1);                                      // Độ dày 1 pixel
    }

    // === Bước 6: Vẽ con trỏ tam giác chỉ altitude hiện tại ===
    std::vector<cv::Point> pointer_pts;  // Vector chứa 3 đỉnh tam giác con trỏ
    pointer_pts.push_back(cv::Point(bar_x - 8, pointer_y));         // Mũi nhọn bên trái - chỉ vào thanh
    pointer_pts.push_back(cv::Point(bar_x - 18, pointer_y - 6));    // Đỉnh trên trái
    pointer_pts.push_back(cv::Point(bar_x - 18, pointer_y + 6));    // Đỉnh dưới trái

    cv::fillConvexPoly(frame,            // Vẽ tam giác tô đặc lên frame
                       pointer_pts,      // 3 đỉnh tam giác
                       cv::Scalar(0, 255, 0)); // Màu xanh lá - đồng bộ với HUD

    // === Bước 7: Hiển thị giá trị altitude chính xác bằng số bên cạnh con trỏ ===
    std::ostringstream alt_text;                              // Tạo string stream cho format
    alt_text << std::fixed << std::setprecision(1);           // 1 chữ số thập phân
    alt_text << hud_data_.altitude << "m";                    // Thêm đơn vị "m" (mét)

    cv::putText(frame,                                        // Vẽ lên frame
                alt_text.str(),                               // Chuỗi altitude (ví dụ: "25.3m")
                cv::Point(bar_x + bar_width + 5, pointer_y + 5), // Bên phải thanh, canh ngang với con trỏ
                cv::FONT_HERSHEY_SIMPLEX,                     // Font đơn giản
                0.45,                                         // Kích thước font nhỏ
                cv::Scalar(0, 255, 0),                        // Màu xanh lá
                1);                                           // Độ dày 1 pixel

    // === Bước 8: Vẽ nhãn "ALT" ở trên đỉnh thanh ===
    cv::putText(frame,                                        // Vẽ lên frame
                "ALT",                                        // Nhãn tiêu đề thanh
                cv::Point(bar_x - 2, bar_top - 8),            // Phía trên thanh, canh giữa
                cv::FONT_HERSHEY_SIMPLEX,                     // Font đơn giản
                0.45,                                         // Kích thước font nhỏ
                cv::Scalar(0, 255, 0),                        // Màu xanh lá
                1);                                           // Độ dày 1 pixel
}

// ============================================================================
// VẼ THANH TRẠNG THÁI - STATUS BAR Ở DƯỚI CÙNG
// ============================================================================

/// @brief Vẽ thanh trạng thái ngang ở dưới cùng khung hình
/// Hiển thị: pin, GPS, chế độ bay trên nền bán trong suốt
void DroneHUD::drawStatusBar(cv::Mat& frame) {
    // Định nghĩa kích thước thanh trạng thái
    int bar_height = 40;                           // Chiều cao thanh = 40 pixel
    int bar_y = frame_height_ - bar_height;        // Vị trí Y = sát mép dưới khung hình

    // === Bước 1: Vẽ nền bán trong suốt cho thanh trạng thái ===
    cv::Mat overlay = frame.clone();               // Clone frame cho overlay
    cv::rectangle(overlay,                          // Vẽ hình chữ nhật nền lên overlay
                  cv::Point(0, bar_y),              // Góc trên trái = mép trái, đỉnh thanh
                  cv::Point(frame_width_, frame_height_), // Góc dưới phải = mép phải, đáy frame
                  cv::Scalar(0, 0, 0),              // Màu đen
                  cv::FILLED);                      // Tô đặc

    cv::addWeighted(overlay, 0.6,                   // 60% đen - đậm hơn để text dễ đọc trên nền tối
                    frame, 0.4,                     // 40% ảnh gốc
                    0, frame);                      // Ghi kết quả vào frame

    // Vẽ viền trên của thanh trạng thái
    cv::line(frame,                                 // Vẽ đường ngang phân cách
             cv::Point(0, bar_y),                   // Bắt đầu từ mép trái
             cv::Point(frame_width_, bar_y),        // Kết thúc tại mép phải
             cv::Scalar(0, 255, 0),                 // Màu xanh lá
             1);                                    // Độ dày 1 pixel

    // Tọa độ Y cho dòng text trong thanh trạng thái
    int text_y = bar_y + 26;                        // Canh giữa dọc trong thanh (bar_y + bar_height/2 + offset)

    // === Bước 2: Hiển thị trạng thái pin với màu theo mức ===
    std::ostringstream battery_text;                 // String stream cho format text pin
    battery_text << "BAT: " << hud_data_.battery_percent << "%"; // Tạo chuỗi "BAT: 75%"

    // Chọn màu theo mức pin: xanh (>50%), cam (20-50%), đỏ (<20%)
    cv::Scalar battery_color;                        // Biến lưu màu hiển thị pin
    if (hud_data_.battery_percent > 50) {            // Pin trên 50% = tốt
        battery_color = cv::Scalar(0, 255, 0);       // Xanh lá = an toàn
    } else if (hud_data_.battery_percent > 20) {     // Pin 20-50% = cần chú ý
        battery_color = cv::Scalar(0, 165, 255);     // Cam = cảnh báo
    } else {                                         // Pin dưới 20% = nguy hiểm
        battery_color = cv::Scalar(0, 0, 255);       // Đỏ = cần hạ cánh ngay
    }

    cv::putText(frame,                               // Vẽ text pin lên frame
                battery_text.str(),                  // Chuỗi "BAT: XX%"
                cv::Point(15, text_y),               // Vị trí bên trái thanh, cách mép 15 pixel
                cv::FONT_HERSHEY_SIMPLEX,            // Font đơn giản
                0.5,                                 // Kích thước font
                battery_color,                       // Màu theo mức pin
                1);                                  // Độ dày 1 pixel

    // === Bước 3: Hiển thị tọa độ GPS ===
    std::ostringstream gps_text;                     // String stream cho GPS
    gps_text << std::fixed << std::setprecision(6);  // 6 chữ số thập phân cho GPS (độ chính xác ~10cm)
    gps_text << "GPS: " << hud_data_.gps_latitude << ", " << hud_data_.gps_longitude; // Format: "GPS: lat, lon"

    cv::putText(frame,                               // Vẽ text GPS lên frame
                gps_text.str(),                      // Chuỗi tọa độ GPS đã format
                cv::Point(frame_width_ / 4, text_y), // Vị trí 1/4 chiều ngang - cách pin một khoảng
                cv::FONT_HERSHEY_SIMPLEX,            // Font đơn giản
                0.45,                                // Kích thước font hơi nhỏ (GPS text dài)
                cv::Scalar(255, 255, 0),             // Màu xanh ngọc (cyan BGR)
                1);                                  // Độ dày 1 pixel

    // === Bước 4: Hiển thị chế độ bay ===
    std::string mode_text = "MODE: " + hud_data_.flight_mode; // Tạo chuỗi chế độ bay

    // Chọn màu theo chế độ bay
    cv::Scalar mode_color;                                    // Biến lưu màu mode
    if (hud_data_.flight_mode == "AUTO") {                    // Chế độ tự động = xanh lá
        mode_color = cv::Scalar(0, 255, 0);
    } else if (hud_data_.flight_mode == "LAND" ||             // Chế độ hạ cánh hoặc quay về
               hud_data_.flight_mode == "RTL") {
        mode_color = cv::Scalar(0, 165, 255);                 // Cam - cần chú ý
    } else if (hud_data_.flight_mode == "STABILIZE") {        // Chế độ ổn định = điều khiển tay
        mode_color = cv::Scalar(0, 255, 255);                 // Vàng
    } else {                                                  // Các chế độ khác
        mode_color = cv::Scalar(255, 255, 255);               // Trắng mặc định
    }

    // Đo kích thước text mode để canh phải
    int mode_baseline = 0;                                    // Biến nhận baseline
    cv::Size mode_size = cv::getTextSize(mode_text,           // Đo chuỗi mode
                                         cv::FONT_HERSHEY_SIMPLEX, // Font
                                         0.5,                // Kích thước
                                         1,                  // Độ dày
                                         &mode_baseline);    // Baseline

    cv::putText(frame,                                        // Vẽ text mode lên frame
                mode_text,                                    // Chuỗi "MODE: STABILIZE"
                cv::Point(frame_width_ - mode_size.width - 15, text_y), // Canh phải, cách mép 15px
                cv::FONT_HERSHEY_SIMPLEX,                     // Font đơn giản
                0.5,                                          // Kích thước font
                mode_color,                                   // Màu theo chế độ
                1);                                           // Độ dày 1 pixel
}

// ============================================================================
// VẼ BỘ ĐẾM FPS - ĐỔI MÀU THEO MỨC HIỆU SUẤT
// ============================================================================

/// @brief Vẽ chỉ số FPS ở góc trên trái với màu phản ánh mức hiệu suất
/// Xanh lá >= 30fps, Vàng >= 15fps, Đỏ < 15fps
void DroneHUD::drawFPSCounter(cv::Mat& frame) {
    // Format chuỗi FPS với 1 chữ số thập phân
    std::ostringstream fps_text;                    // String stream cho format FPS
    fps_text << "FPS: " << std::fixed << std::setprecision(1) << hud_data_.fps; // "FPS: 29.8"

    // Chọn màu dựa trên giá trị FPS - phản ánh trực quan mức hiệu suất
    cv::Scalar fps_color;                           // Biến lưu màu FPS
    if (hud_data_.fps >= 30.0) {                    // FPS >= 30 = mượt mà, xử lý tốt
        fps_color = cv::Scalar(0, 255, 0);          // Xanh lá = hiệu suất tốt
    } else if (hud_data_.fps >= 15.0) {             // FPS 15-29 = chấp nhận được nhưng không lý tưởng
        fps_color = cv::Scalar(0, 255, 255);        // Vàng = cần tối ưu
    } else {                                        // FPS < 15 = quá chậm, ảnh hưởng điều khiển
        fps_color = cv::Scalar(0, 0, 255);          // Đỏ = cảnh báo hiệu suất kém
    }

    // Sử dụng hàm tiện ích vẽ text có nền để FPS nổi bật trên mọi ảnh nền
    drone_utils::drawTextWithBackground(
        frame,                                      // Khung hình đích
        fps_text.str(),                             // Chuỗi "FPS: XX.X"
        cv::Point(15, 30),                          // Góc trên trái, cách mép 15px ngang và 30px dọc
        cv::FONT_HERSHEY_SIMPLEX,                   // Font đơn giản
        0.6,                                        // Kích thước font vừa phải
        fps_color,                                  // Màu theo mức FPS
        cv::Scalar(0, 0, 0),                        // Nền đen
        2,                                          // Độ dày text = 2 cho đậm nét
        5);                                         // Padding 5 pixel quanh text
}

// ============================================================================
// VẼ TIMESTAMP - THỜI GIAN HIỆN TẠI Ở GÓC TRÊN PHẢI
// ============================================================================

/// @brief Vẽ nhãn thời gian hiện tại ở góc trên phải theo format HH:MM:SS
/// Sử dụng std::chrono và std::localtime để lấy giờ hệ thống chính xác
void DroneHUD::drawTimestamp(cv::Mat& frame) {
    // Lấy thời gian hiện tại từ system clock
    auto now = std::chrono::system_clock::now();              // Lấy thời điểm hiện tại từ đồng hồ hệ thống
    auto time_t_now = std::chrono::system_clock::to_time_t(now); // Chuyển sang time_t để dùng localtime

    // Chuyển đổi sang cấu trúc tm để trích xuất giờ/phút/giây
    std::tm local_tm;                          // Cấu trúc lưu thời gian local đã phân tách

#ifdef _WIN32                                  // Kiểm tra hệ điều hành Windows
    localtime_s(&local_tm, &time_t_now);       // Dùng localtime_s (thread-safe) trên Windows
#else                                          // Hệ điều hành khác (Linux, macOS)
    localtime_r(&time_t_now, &local_tm);       // Dùng localtime_r (thread-safe) trên POSIX
#endif                                         // Kết thúc khối điều kiện biên dịch

    // Format thời gian thành chuỗi "HH:MM:SS"
    std::ostringstream time_text;              // String stream cho format thời gian
    time_text << std::setfill('0')             // Điền số 0 trước các số < 10 (ví dụ: 09 thay vì 9)
              << std::setw(2) << local_tm.tm_hour << ":"  // Giờ 2 chữ số + dấu ":"
              << std::setw(2) << local_tm.tm_min << ":"   // Phút 2 chữ số + dấu ":"
              << std::setw(2) << local_tm.tm_sec;         // Giây 2 chữ số

    // Đo kích thước text để canh phải chính xác
    int baseline = 0;                          // Biến nhận baseline
    cv::Size text_size = cv::getTextSize(      // Đo kích thước pixel của chuỗi thời gian
        time_text.str(),                       // Chuỗi cần đo
        cv::FONT_HERSHEY_SIMPLEX,              // Font chữ
        0.6,                                   // Kích thước font
        1,                                     // Độ dày nét
        &baseline);                            // Nhận baseline

    // Vẽ timestamp có nền ở góc trên phải
    drone_utils::drawTextWithBackground(
        frame,                                              // Khung hình đích
        time_text.str(),                                    // Chuỗi "HH:MM:SS"
        cv::Point(frame_width_ - text_size.width - 20, 30), // Canh phải, cách mép 20px, dòng trên cùng
        cv::FONT_HERSHEY_SIMPLEX,                           // Font đơn giản
        0.6,                                                // Kích thước font
        cv::Scalar(255, 255, 255),                          // Màu trắng cho timestamp
        cv::Scalar(0, 0, 0),                                // Nền đen
        1,                                                  // Độ dày text
        5);                                                 // Padding 5 pixel
}

// ============================================================================
// VẼ CẢNH BÁO - THÔNG ĐIỆP NHẤP NHÁY MÀU ĐỎ
// ============================================================================

/// @brief Vẽ thông điệp cảnh báo nhấp nháy ở giữa phía trên khung hình
/// Hiệu ứng nhấp nháy: bật/tắt mỗi 500ms dựa trên thời gian thực
void DroneHUD::drawWarning(cv::Mat& frame) {
    // Không vẽ gì nếu không có thông điệp cảnh báo
    if (hud_data_.warning_message.empty()) {    // Kiểm tra chuỗi cảnh báo có rỗng không
        return;                                 // Thoát ngay - không có cảnh báo để hiển thị
    }

    // Tính toán hiệu ứng nhấp nháy dựa trên thời gian thực
    auto now = std::chrono::steady_clock::now();  // Lấy thời điểm hiện tại (steady_clock chính xác cho đo khoảng thời gian)

    // Tính khoảng thời gian kể từ lần toggle cuối
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( // Chuyển đổi sang mili giây
        now - last_warning_toggle_).count();                              // count() lấy giá trị số nguyên ms

    // Toggle trạng thái hiện/ẩn mỗi 500 mili giây (2 lần nhấp nháy mỗi giây)
    if (elapsed >= 500) {                         // Đã qua 500ms kể từ lần toggle cuối
        warning_visible_ = !warning_visible_;     // Đảo trạng thái: hiện -> ẩn hoặc ẩn -> hiện
        last_warning_toggle_ = now;               // Cập nhật thời điểm toggle mới
    }

    // Chỉ vẽ khi đang ở trạng thái "hiện" của chu kỳ nhấp nháy
    if (!warning_visible_) {                      // Nếu đang ở pha "ẩn"
        return;                                   // Không vẽ gì - tạo hiệu ứng nhấp nháy
    }

    // Đo kích thước text cảnh báo để canh giữa
    int baseline = 0;                             // Biến nhận baseline
    cv::Size text_size = cv::getTextSize(         // Đo kích thước pixel
        hud_data_.warning_message,                // Chuỗi cảnh báo
        cv::FONT_HERSHEY_SIMPLEX,                 // Font
        0.8,                                      // Kích thước font lớn hơn bình thường - thu hút chú ý
        2,                                        // Độ dày nét đậm
        &baseline);                               // Nhận baseline

    // Tính vị trí canh giữa theo chiều ngang, đặt ở 1/4 phía trên khung hình
    int text_x = (frame_width_ - text_size.width) / 2;   // X canh giữa = (width - text_width) / 2
    int text_y = frame_height_ / 4;                       // Y = 1/4 chiều cao - vùng dễ chú ý

    // Vẽ nền đỏ bán trong suốt phía sau text cảnh báo
    cv::Mat overlay = frame.clone();              // Clone frame cho overlay
    cv::rectangle(overlay,                        // Vẽ nền đỏ lên overlay
                  cv::Point(text_x - 15, text_y - text_size.height - 10), // Góc trên trái (có padding)
                  cv::Point(text_x + text_size.width + 15, text_y + 10),  // Góc dưới phải (có padding)
                  cv::Scalar(0, 0, 150),          // Màu đỏ đậm cho nền
                  cv::FILLED);                    // Tô đặc

    cv::addWeighted(overlay, 0.5,                 // 50% overlay đỏ
                    frame, 0.5,                   // 50% ảnh gốc
                    0, frame);                    // Kết quả ghi vào frame

    // Vẽ viền trắng quanh khung cảnh báo
    cv::rectangle(frame,                          // Vẽ viền trực tiếp lên frame đã blend
                  cv::Point(text_x - 15, text_y - text_size.height - 10), // Cùng vị trí với nền
                  cv::Point(text_x + text_size.width + 15, text_y + 10),  // Cùng kích thước
                  cv::Scalar(255, 255, 255),      // Viền trắng
                  2);                             // Độ dày viền = 2 pixel

    // Vẽ ký hiệu "⚠" (dấu cảnh báo) bằng text đơn giản trước thông điệp
    std::string display_text = "! " + hud_data_.warning_message + " !"; // Thêm dấu ! hai bên

    // Vẽ text cảnh báo màu trắng sáng trên nền đỏ
    cv::putText(frame,                            // Vẽ text cảnh báo lên frame
                display_text,                     // Chuỗi "! CẢNH BÁO !"
                cv::Point(text_x - 10, text_y),   // Vị trí đã canh giữa
                cv::FONT_HERSHEY_SIMPLEX,         // Font đơn giản
                0.8,                              // Kích thước font lớn
                cv::Scalar(255, 255, 255),        // Màu trắng nổi bật trên nền đỏ
                2);                               // Độ dày đậm = 2 pixel
}

// ============================================================================
// VẼ TÓM TẮT NHẬN DIỆN - SỐ ĐỐI TƯỢNG PHÁT HIỆN ĐƯỢC
// ============================================================================

/// @brief Vẽ số lượng đối tượng phát hiện được ở góc trên trái (dưới FPS)
/// Hiển thị kết quả từ module nhận diện đối tượng hoặc ArUco marker
void DroneHUD::drawDetectionSummary(cv::Mat& frame) {
    // Tạo chuỗi hiển thị số đối tượng
    std::ostringstream det_text;                   // String stream cho format
    det_text << "DET: " << hud_data_.detection_count; // "DET: 3" - viết tắt "Detections"

    // Chọn màu: xanh lá nếu có detection, xám nếu không có
    cv::Scalar det_color = (hud_data_.detection_count > 0)  // Kiểm tra có detection không
                           ? cv::Scalar(0, 255, 0)          // Xanh lá = đang phát hiện đối tượng
                           : cv::Scalar(150, 150, 150);     // Xám = không phát hiện gì

    // Vẽ text detection có nền, đặt dưới FPS counter
    drone_utils::drawTextWithBackground(
        frame,                                     // Khung hình đích
        det_text.str(),                            // Chuỗi "DET: X"
        cv::Point(15, 58),                         // Dưới FPS (FPS ở Y=30), cách 28 pixel
        cv::FONT_HERSHEY_SIMPLEX,                  // Font đơn giản
        0.5,                                       // Kích thước font hơi nhỏ hơn FPS
        det_color,                                 // Màu theo trạng thái detection
        cv::Scalar(0, 0, 0),                       // Nền đen
        1,                                         // Độ dày text
        4);                                        // Padding 4 pixel
}
