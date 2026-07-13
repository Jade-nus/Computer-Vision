// ============================================================================
// FILE: src/utils.cpp
// PROJECT: DroneVisionPro - Hệ thống thị giác máy tính cho drone
// PURPOSE: Triển khai đầy đủ các tiện ích chung - FPSCounter, logging, vẽ text
// AUTHOR: Tran Ngoc Bao - 24021238
// DATE: 2026-07-13
// DESCRIPTION: File này chứa implementation cho tất cả hàm và lớp tiện ích
//              bao gồm: FPSCounter sliding window, camera calibration mặc định,
//              console logging có timestamp, và vẽ text có nền
// ============================================================================

#include "utils.hpp" // Include header tiện ích - chứa khai báo lớp FPSCounter, hàm và namespace

/// @brief Mở namespace drone_utils - tất cả implementation nằm trong namespace này
namespace drone_utils {

// ============================================================================
// LỚP FPSCounter - TRIỂN KHAI BỘ ĐẾM FPS BẰNG CỬA SỔ TRƯỢT
// ============================================================================

/// @brief Constructor - khởi tạo FPSCounter với kích thước cửa sổ trượt
/// Cửa sổ trượt (sliding window) lưu N timestamp gần nhất để tính FPS trung bình
/// Kích thước lớn hơn cho kết quả ổn định hơn nhưng phản ứng chậm hơn với thay đổi
FPSCounter::FPSCounter(size_t window_size)
    : window_size_(window_size) // Gán kích thước cửa sổ - mặc định 30 frame cho kết quả mượt
{
    // Deque timestamps_ được khởi tạo rỗng bởi constructor mặc định của std::deque
    // Không cần khởi tạo thêm - sẽ được điền dần khi gọi tick()
}

/// @brief Ghi nhận một tick - thêm timestamp hiện tại vào cửa sổ trượt
/// Gọi phương thức này MỘT LẦN cho mỗi frame được xử lý để đo FPS chính xác
/// Sử dụng steady_clock vì nó đơn điệu tăng - không bị ảnh hưởng bởi thay đổi giờ hệ thống
void FPSCounter::tick() {
    // Lấy thời điểm hiện tại bằng steady_clock - đồng hồ ổn định không bị điều chỉnh
    auto now = std::chrono::steady_clock::now(); // steady_clock phù hợp cho đo khoảng thời gian

    // Thêm timestamp mới vào cuối deque
    timestamps_.push_back(now); // push_back thêm phần tử ở cuối - O(1) amortized

    // Nếu deque đã vượt quá kích thước cửa sổ, loại bỏ timestamp cũ nhất ở đầu
    while (timestamps_.size() > window_size_) { // Kiểm tra kích thước vượt quá giới hạn
        timestamps_.pop_front();                 // Loại bỏ phần tử đầu tiên (cũ nhất) - O(1)
    }
    // Sau vòng lặp, deque luôn có tối đa window_size_ phần tử
}

/// @brief Tính FPS hiện tại từ cửa sổ trượt timestamp
/// Công thức: FPS = (số_frame - 1) / tổng_thời_gian_giữa_frame_đầu_và_cuối
/// Cần ít nhất 2 timestamp để tính được FPS (1 khoảng thời gian)
/// @return Giá trị FPS trung bình, hoặc 0.0 nếu chưa đủ dữ liệu
double FPSCounter::getFPS() const {
    // Kiểm tra có đủ dữ liệu để tính FPS không - cần tối thiểu 2 mẫu
    if (timestamps_.size() < 2) {    // Nếu có ít hơn 2 timestamp
        return 0.0;                  // Trả về 0 - chưa đủ dữ liệu để tính
    }

    // Tính tổng thời gian giữa timestamp đầu tiên và cuối cùng trong cửa sổ
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( // Chuyển khoảng thời gian sang ms
        timestamps_.back() - timestamps_.front() // back() = mới nhất, front() = cũ nhất
    ).count();                                   // count() trả về giá trị số nguyên mili giây

    // Tránh chia cho 0 trong trường hợp tất cả timestamp giống nhau (xử lý quá nhanh)
    if (duration == 0) {             // Nếu khoảng thời gian = 0ms (không thể thực tế nhưng phòng bị)
        return 0.0;                  // Trả về 0 để tránh lỗi chia cho 0
    }

    // Tính FPS: số khoảng thời gian = (số mẫu - 1), chia cho tổng thời gian (giây)
    // Nhân 1000 để chuyển ms sang giây: FPS = (N-1) * 1000 / duration_ms
    double fps = static_cast<double>(timestamps_.size() - 1) * 1000.0 // Số khoảng * 1000 (chuyển sang giây)
                 / static_cast<double>(duration);                      // Chia cho tổng thời gian ms
    return fps; // Trả về FPS trung bình của cửa sổ trượt
}

// ============================================================================
// HÀM getDefaultCameraMatrix - TẠO MA TRẬN CAMERA MẶC ĐỊNH
// ============================================================================

/// @brief Tạo ma trận camera nội tại mặc định với các giá trị hợp lý cho drone
/// Ma trận camera (intrinsic matrix) mô tả đặc tính quang học của camera:
///   [fx  0  cx]     fx, fy = tiêu cự (focal length) tính bằng pixel
///   [0  fy  cy]     cx, cy = điểm chính (principal point) - thường ở tâm ảnh
///   [0   0   1]     Dòng cuối cố định cho tọa độ thuần nhất
/// Giá trị mặc định: fx = fy = frame_width (giả định FOV ~53° ngang)
///                    cx = width/2, cy = height/2 (tâm ảnh)
CameraCalibration getDefaultCameraMatrix(int frame_width, int frame_height) {
    CameraCalibration calib;  // Tạo đối tượng CameraCalibration với is_calibrated = false mặc định

    // Tính tiêu cự mặc định = chiều rộng frame
    // Lý do: với sensor 35mm, focal length = width pixel cho FOV khoảng 53 độ
    // Đây là giá trị hợp lý cho camera drone góc rộng thông dụng
    double focal_length = static_cast<double>(frame_width); // Focal length = chiều rộng frame (pixel)

    // Tính điểm chính (principal point) = tâm ảnh
    // Giả định tâm quang học trùng với tâm hình học - đúng cho hầu hết camera
    double cx = static_cast<double>(frame_width) / 2.0;   // Tọa độ X tâm = nửa chiều rộng
    double cy = static_cast<double>(frame_height) / 2.0;  // Tọa độ Y tâm = nửa chiều cao

    // Tạo ma trận camera 3x3 kiểu CV_64F (double precision)
    calib.camera_matrix = cv::Mat::eye(3, 3, CV_64F); // Khởi tạo ma trận đơn vị 3x3 - đường chéo = 1

    // Gán các giá trị vào ma trận camera
    calib.camera_matrix.at<double>(0, 0) = focal_length; // fx - tiêu cự theo trục X (hàng 0, cột 0)
    calib.camera_matrix.at<double>(1, 1) = focal_length; // fy - tiêu cự theo trục Y (hàng 1, cột 1)
    calib.camera_matrix.at<double>(0, 2) = cx;           // cx - tọa độ X điểm chính (hàng 0, cột 2)
    calib.camera_matrix.at<double>(1, 2) = cy;           // cy - tọa độ Y điểm chính (hàng 1, cột 2)
    // Giá trị (2,2) = 1.0 đã được đặt bởi Mat::eye - hệ số tọa độ thuần nhất

    // Tạo vector hệ số méo = 0 (giả định không méo)
    // Vector 5 phần tử: [k1, k2, p1, p2, k3] - tất cả = 0 nghĩa là không méo
    calib.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F); // Ma trận 5x1 toàn số 0 - không méo

    // Đánh dấu đây KHÔNG phải calibration chính xác - chỉ là giá trị ước lượng
    calib.is_calibrated = false; // false vì chưa qua quy trình calibration thực tế với bàn cờ

    return calib; // Trả về cấu trúc CameraCalibration với ma trận mặc định
}

// ============================================================================
// HÀM createDirectory - TẠO THƯ MỤC ĐỆ QUY
// ============================================================================

/// @brief Tạo thư mục và tất cả thư mục cha cần thiết nếu chưa tồn tại
/// Sử dụng std::filesystem của C++17 - tự động tạo đệ quy các thư mục trung gian
/// Ví dụ: createDirectory("a/b/c") sẽ tạo cả "a", "a/b", "a/b/c" nếu chưa có
bool createDirectory(const std::string& path) {
    try {                                       // Bắt ngoại lệ filesystem có thể xảy ra
        // create_directories tạo thư mục đệ quy - trả về true nếu tạo mới
        // Nếu thư mục đã tồn tại, trả về false nhưng KHÔNG báo lỗi
        std::filesystem::create_directories(path); // Tạo thư mục và các thư mục cha

        return true;                             // Thành công - thư mục đã tồn tại hoặc vừa được tạo
    }
    catch (const std::filesystem::filesystem_error& e) { // Bắt lỗi filesystem cụ thể
        // In thông báo lỗi ra console - dùng cerr cho error stream
        std::cerr << "[ERROR] Khong the tao thu muc '" << path // Thông báo lỗi tiếng Việt không dấu
                  << "': " << e.what() << std::endl;            // Chi tiết lỗi từ exception
        return false;                            // Thất bại - trả về false
    }
}

// ============================================================================
// HÀM getCurrentTimestamp - LẤY CHUỖI THỜI GIAN HIỆN TẠI
// ============================================================================

/// @brief Lấy timestamp hiện tại dưới dạng chuỗi "YYYY-MM-DD HH:MM:SS"
/// Dùng cho logging, đặt tên file, và hiển thị trên HUD
/// Sử dụng std::chrono + std::localtime để lấy giờ hệ thống chính xác
std::string getCurrentTimestamp() {
    // Lấy thời gian hiện tại từ system_clock (đồng hồ hệ thống - gắn với giờ thực)
    auto now = std::chrono::system_clock::now();              // Thời điểm hiện tại
    auto time_t_now = std::chrono::system_clock::to_time_t(now); // Chuyển sang time_t cổ điển

    // Chuyển đổi sang cấu trúc tm để trích xuất các thành phần thời gian
    std::tm local_tm;                             // Cấu trúc lưu năm/tháng/ngày/giờ/phút/giây

#ifdef _WIN32                                     // Kiểm tra biên dịch trên Windows
    localtime_s(&local_tm, &time_t_now);          // Phiên bản thread-safe trên MSVC
#else                                             // Linux/macOS
    localtime_r(&time_t_now, &local_tm);          // Phiên bản thread-safe trên POSIX
#endif                                            // Kết thúc khối điều kiện biên dịch

    // Format thời gian thành chuỗi "YYYY-MM-DD HH:MM:SS"
    std::ostringstream oss;                        // String stream cho format phức tạp
    oss << (1900 + local_tm.tm_year) << "-"        // Năm = tm_year + 1900 (tm_year tính từ 1900)
        << std::setfill('0') << std::setw(2) << (1 + local_tm.tm_mon) << "-"  // Tháng = tm_mon + 1 (0-indexed)
        << std::setw(2) << local_tm.tm_mday << " " // Ngày trong tháng (1-31)
        << std::setw(2) << local_tm.tm_hour << ":"  // Giờ (0-23)
        << std::setw(2) << local_tm.tm_min << ":"   // Phút (0-59)
        << std::setw(2) << local_tm.tm_sec;         // Giây (0-59)

    return oss.str(); // Trả về chuỗi timestamp đã format hoàn chỉnh
}

// ============================================================================
// CÁC HÀM LOGGING - GHI LOG CÓ MÀU VÀ TIMESTAMP
// ============================================================================

/// @brief Ghi log thông tin [INFO] ra console với timestamp
/// Tiền tố xanh lá [INFO] giúp phân biệt nhanh loại log khi đọc console
/// @param msg Nội dung thông điệp cần ghi
void logInfo(const std::string& msg) {
    // Lấy timestamp hiện tại cho log entry
    std::string timestamp = getCurrentTimestamp(); // Format: "YYYY-MM-DD HH:MM:SS"

    // In ra console với format: [YYYY-MM-DD HH:MM:SS] [INFO] message
    // Sử dụng cout (standard output) cho thông tin bình thường
    std::cout << "[" << timestamp << "] "          // Timestamp trong ngoặc vuông
              << "[INFO] "                          // Tiền tố loại log
              << msg                                // Nội dung thông điệp
              << std::endl;                         // Xuống dòng và flush buffer ngay lập tức
}

/// @brief Ghi log cảnh báo [WARN] ra console với timestamp
/// Tiền tố vàng [WARN] cho các tình huống bất thường nhưng không phải lỗi
/// @param msg Nội dung cảnh báo cần ghi
void logWarning(const std::string& msg) {
    // Lấy timestamp cho log entry
    std::string timestamp = getCurrentTimestamp(); // Format: "YYYY-MM-DD HH:MM:SS"

    // In ra cout với format chuẩn - [WARN] dễ grep trong log file
    std::cout << "[" << timestamp << "] "          // Timestamp
              << "[WARN] "                          // Tiền tố cảnh báo
              << msg                                // Nội dung cảnh báo
              << std::endl;                         // Xuống dòng + flush
}

/// @brief Ghi log lỗi [ERROR] ra error stream với timestamp
/// Tiền tố đỏ [ERROR] cho các lỗi nghiêm trọng cần xử lý
/// Dùng cerr thay vì cout - ghi vào stderr để có thể tách riêng khi redirect output
/// @param msg Nội dung lỗi cần ghi
void logError(const std::string& msg) {
    // Lấy timestamp cho log entry
    std::string timestamp = getCurrentTimestamp(); // Format: "YYYY-MM-DD HH:MM:SS"

    // In ra cerr (error stream) - tách biệt với output bình thường
    // Khi chạy: program.exe > output.log 2> error.log - lỗi sẽ vào file riêng
    std::cerr << "[" << timestamp << "] "          // Timestamp
              << "[ERROR] "                         // Tiền tố lỗi
              << msg                                // Nội dung lỗi
              << std::endl;                         // Xuống dòng + flush
}

// ============================================================================
// HÀM drawTextWithBackground - VẼ TEXT CÓ NỀN HÌNH CHỮ NHẬT
// ============================================================================

/// @brief Vẽ text với nền hình chữ nhật đặc phía sau để text luôn dễ đọc
/// Quy trình: đo kích thước text -> vẽ hình chữ nhật nền -> vẽ text lên trên
/// Kỹ thuật này đảm bảo text nổi bật trên bất kỳ ảnh nền phức tạp nào
void drawTextWithBackground(
    cv::Mat& frame,            // Khung hình đích - sẽ được vẽ trực tiếp
    const std::string& text,   // Chuỗi text cần hiển thị
    cv::Point pos,             // Vị trí baseline (dòng cơ sở) của text
    int font_face,             // Loại font OpenCV (FONT_HERSHEY_SIMPLEX, etc.)
    double font_scale,         // Tỷ lệ kích thước font (1.0 = kích thước chuẩn)
    cv::Scalar text_color,     // Màu text BGR (Blue, Green, Red)
    cv::Scalar bg_color,       // Màu nền BGR cho hình chữ nhật
    int thickness,             // Độ dày nét vẽ text (pixel)
    int padding)               // Khoảng đệm giữa text và viền nền (pixel)
{
    // === Bước 1: Đo kích thước text bằng getTextSize ===
    // getTextSize trả về kích thước bounding box của text khi render với font cụ thể
    int baseline = 0;                              // Biến nhận baseline - khoảng cách từ dòng cơ sở đến đáy ký tự (ví dụ: phần đuôi chữ 'g', 'y')

    cv::Size text_size = cv::getTextSize(          // Hàm đo kích thước text pixel
        text,                                      // Chuỗi cần đo
        font_face,                                 // Kiểu font - ảnh hưởng kích thước ký tự
        font_scale,                                // Tỷ lệ phóng - nhân với kích thước cơ bản
        thickness,                                 // Độ dày nét - nét dày hơn = text rộng hơn
        &baseline);                                // Con trỏ nhận giá trị baseline

    // Thêm thickness vào baseline để padding chính xác hơn
    baseline += thickness;                         // Bổ sung độ dày nét vào baseline

    // === Bước 2: Tính tọa độ hình chữ nhật nền bao quanh text ===
    // pos là vị trí baseline (đáy) của text, nên top = pos.y - text_height
    cv::Point rect_top_left(
        pos.x - padding,                           // X trái = vị trí text - padding
        pos.y - text_size.height - padding          // Y trên = baseline - chiều cao text - padding
    );

    cv::Point rect_bottom_right(
        pos.x + text_size.width + padding,          // X phải = vị trí text + chiều rộng text + padding
        pos.y + baseline + padding                  // Y dưới = baseline + baseline offset + padding
    );

    // === Bước 3: Đảm bảo hình chữ nhật nằm trong phạm vi frame ===
    // Clamp tọa độ để tránh vẽ ra ngoài ảnh gây lỗi
    rect_top_left.x = std::max(0, rect_top_left.x);                    // X trái không nhỏ hơn 0
    rect_top_left.y = std::max(0, rect_top_left.y);                    // Y trên không nhỏ hơn 0
    rect_bottom_right.x = std::min(frame.cols, rect_bottom_right.x);   // X phải không vượt chiều rộng
    rect_bottom_right.y = std::min(frame.rows, rect_bottom_right.y);   // Y dưới không vượt chiều cao

    // === Bước 4: Vẽ hình chữ nhật nền đặc ===
    cv::rectangle(frame,                           // Vẽ trực tiếp lên frame
                  rect_top_left,                   // Góc trên trái đã clamp
                  rect_bottom_right,               // Góc dưới phải đã clamp
                  bg_color,                        // Màu nền (mặc định đen)
                  cv::FILLED);                     // FILLED = tô đặc toàn bộ hình chữ nhật

    // === Bước 5: Vẽ text lên trên hình chữ nhật nền ===
    cv::putText(frame,                             // Vẽ lên frame (trên nền đã vẽ)
                text,                              // Chuỗi text cần hiển thị
                pos,                               // Vị trí baseline gốc
                font_face,                         // Kiểu font
                font_scale,                        // Tỷ lệ kích thước
                text_color,                        // Màu text (nổi bật trên nền)
                thickness);                        // Độ dày nét vẽ
}

} // Kết thúc namespace drone_utils - đóng phạm vi namespace
