/**
 * @file qr_detector.hpp
 * @brief Module phát hiện và giải mã QR Code cho hệ thống DroneVisionPro
 * @details File header định nghĩa lớp QRDetector với đầy đủ chức năng:
 *          - Phát hiện và giải mã QR code đơn lẻ và nhiều QR code
 *          - Vẽ kết quả phát hiện lên khung hình (bounding box + text)
 *          - Lưu lịch sử quét QR code với timestamp
 *          - Phân tích dữ liệu QR code thành lệnh điều khiển drone
 *          Hỗ trợ các định dạng lệnh:
 *          - WAYPOINT:lat,lon,alt  (điểm bay đến)
 *          - ACTION:takeoff/land/hover (hành động)
 *          - URL:https://...       (đường dẫn web)
 *          - CONFIG:key=value      (cấu hình)
 * @author Tran Ngoc Bao - 24021238
 * @date 2026-07-13
 */

#pragma once // Đảm bảo file header chỉ được include một lần duy nhất, tránh lỗi multiple definition

// ============================================================================
// PHẦN INCLUDE CÁC THƯ VIỆN CẦN THIẾT
// ============================================================================

#include <opencv2/opencv.hpp>          // Thư viện OpenCV chính (Mat, Rect, Point2f, putText, ...)
#include <opencv2/objdetect.hpp>       // Module object detection chứa QRCodeDetector của OpenCV

#include <vector>                      // Container vector của STL (danh sách QRInfo, decoded data, ...)
#include <string>                      // Kiểu chuỗi string (dữ liệu QR, tên lệnh, ...)
#include <map>                         // Container map (ánh xạ key-value cho params lệnh drone)
#include <chrono>                      // Thư viện thời gian C++11+ (time_point, system_clock, ...)
#include <deque>                       // Container deque (hàng đợi hai đầu) cho lịch sử QR code
#include <mutex>                       // Mutex để đồng bộ hóa truy cập lịch sử trong đa luồng

// ============================================================================
// ĐỊNH NGHĨA NAMESPACE VÀ CẤU TRÚC DỮ LIỆU
// ============================================================================

namespace drone_vision { // Namespace chính của dự án DroneVisionPro

/**
 * @struct QRInfo
 * @brief Cấu trúc lưu trữ thông tin đầy đủ của một QR code đã phát hiện
 * @details Mỗi QR code phát hiện và giải mã thành công sẽ có một QRInfo chứa:
 *          - Dữ liệu đã giải mã (text, URL, lệnh drone, ...)
 *          - Tọa độ 4 góc trên ảnh 2D
 *          - Hình chữ nhật bao quanh (bounding box)
 *          - Thời điểm phát hiện (timestamp)
 */
struct QRInfo {
    std::string data;                  // Dữ liệu đã giải mã từ QR code (chuỗi text chứa trong QR)

    std::vector<cv::Point2f> points;   // Vector chứa tọa độ 4 góc của QR code trên ảnh 2D
                                       // Thứ tự: top-left, top-right, bottom-right, bottom-left

    cv::Rect bbox;                     // Hình chữ nhật bao quanh QR code (bounding box)
                                       // Dùng để vẽ khung và xác định vùng chứa QR

    std::chrono::system_clock::time_point timestamp; // Thời điểm phát hiện QR code
                                                      // Dùng system_clock để có thể chuyển đổi sang wall-clock time

    /**
     * @brief Constructor mặc định khởi tạo QRInfo
     * @details Khởi tạo các trường với giá trị mặc định an toàn:
     *          - data rỗng, bbox (0,0,0,0), timestamp = thời điểm hiện tại
     */
    QRInfo()                           // Constructor mặc định
        : data("")                     // Dữ liệu rỗng (chưa giải mã)
        , bbox(0, 0, 0, 0)            // Bounding box rỗng (vị trí 0, kích thước 0)
        , timestamp(std::chrono::system_clock::now()) // Gán timestamp = thời điểm tạo đối tượng
    {} // Kết thúc constructor mặc định
};

/**
 * @struct DroneCommand
 * @brief Cấu trúc biểu diễn lệnh điều khiển drone được trích xuất từ QR code
 * @details Lệnh drone có định dạng: TYPE:param1,param2,...
 *          Các loại lệnh hỗ trợ:
 *          - WAYPOINT: điểm bay đến (params: lat, lon, alt)
 *          - ACTION: hành động (params: action = takeoff/land/hover/return)
 *          - URL: đường dẫn web (params: url = https://...)
 *          - CONFIG: cấu hình (params: key=value pairs)
 *          - UNKNOWN: lệnh không nhận dạng được
 */
struct DroneCommand {
    std::string type;                  // Loại lệnh: "WAYPOINT", "ACTION", "URL", "CONFIG", "UNKNOWN"

    std::map<std::string, std::string> params; // Bảng ánh xạ tham số: key -> value
                                                // Ví dụ: {"lat": "21.0285", "lon": "105.8542", "alt": "100"}

    std::string raw_data;              // Dữ liệu gốc từ QR code (trước khi parse), dùng để debug

    bool valid;                        // Cờ cho biết lệnh có hợp lệ không (parse thành công)

    /**
     * @brief Constructor mặc định khởi tạo DroneCommand
     * @details Khởi tạo lệnh với type = "UNKNOWN" và valid = false
     */
    DroneCommand()                     // Constructor mặc định
        : type("UNKNOWN")             // Loại mặc định là UNKNOWN (chưa xác định)
        , raw_data("")                // Dữ liệu gốc rỗng
        , valid(false)                // Lệnh chưa hợp lệ
    {} // Kết thúc constructor mặc định
};

// ============================================================================
// ĐỊNH NGHĨA LỚP CHÍNH QRDETECTOR
// ============================================================================

/**
 * @class QRDetector
 * @brief Lớp phát hiện, giải mã QR code và phân tích lệnh drone
 * @details Pipeline xử lý QR code:
 *          1. Phát hiện QR code trong khung hình (detect / detectMulti)
 *          2. Giải mã nội dung (decode)
 *          3. Lưu vào lịch sử với timestamp
 *          4. Phân tích thành lệnh drone (parseCommand)
 *          5. Vẽ kết quả trực quan (drawDetections)
 *
 *          Lớp thread-safe: sử dụng mutex bảo vệ lịch sử quét
 */
class QRDetector {
public: // Phần public - giao diện sử dụng từ bên ngoài

    // ========================================================================
    // CONSTRUCTOR VÀ DESTRUCTOR
    // ========================================================================

    /**
     * @brief Constructor mặc định của QRDetector
     * @details Khởi tạo cv::QRCodeDetector nội bộ và thiết lập các biến thành viên.
     *          Giới hạn lịch sử mặc định = 100 mục (tránh tràn bộ nhớ).
     */
    QRDetector(); // Constructor không tham số

    /**
     * @brief Destructor mặc định
     * @details Sử dụng destructor tự động vì tất cả tài nguyên
     *          đều được quản lý bởi RAII (vector, deque, ...)
     */
    ~QRDetector() = default; // Destructor mặc định do compiler tạo

    // ========================================================================
    // CÁC PHƯƠNG THỨC PHÁT HIỆN (DETECTION METHODS)
    // ========================================================================

    /**
     * @brief Phát hiện và giải mã một QR code trong khung hình
     * @param frame Khung hình đầu vào từ camera (cv::Mat, BGR hoặc grayscale)
     * @return Vector chứa QRInfo (0 hoặc 1 phần tử) - rỗng nếu không tìm thấy
     * @details Sử dụng cv::QRCodeDetector::detectAndDecode để phát hiện
     *          QR code duy nhất gần nhất/rõ nhất trong ảnh.
     *          Kết quả được lưu vào detected_qrs_ và thêm vào lịch sử.
     */
    std::vector<QRInfo> detect(const cv::Mat& frame); // Phát hiện 1 QR code

    /**
     * @brief Phát hiện và giải mã nhiều QR code trong khung hình
     * @param frame Khung hình đầu vào từ camera (cv::Mat, BGR hoặc grayscale)
     * @return Vector chứa QRInfo của tất cả QR code phát hiện được
     * @details Sử dụng cv::QRCodeDetector::detectAndDecodeMulti (OpenCV 4.x)
     *          để phát hiện đồng thời nhiều QR code. Nếu API không khả dụng,
     *          fallback về detect() đơn lẻ.
     */
    std::vector<QRInfo> detectMulti(const cv::Mat& frame); // Phát hiện nhiều QR code cùng lúc

    // ========================================================================
    // CÁC PHƯƠNG THỨC VẼ KẾT QUẢ (VISUALIZATION METHODS)
    // ========================================================================

    /**
     * @brief Vẽ kết quả phát hiện QR code lên khung hình
     * @param frame Khung hình đầu ra (cv::Mat, pass by reference, vẽ trực tiếp)
     * @details Vẽ cho mỗi QR code:
     *          - Bounding box (hình chữ nhật bao quanh) màu xanh
     *          - Dữ liệu đã giải mã (text) phía trên bounding box
     *          - Đường nối 4 góc QR code
     *          - Thống kê tổng số QR code góc trên-trái ảnh
     */
    void drawDetections(cv::Mat& frame); // Vẽ kết quả lên frame

    // ========================================================================
    // CÁC PHƯƠNG THỨC TRUY VẤN DỮ LIỆU (DATA ACCESS METHODS)
    // ========================================================================

    /**
     * @brief Lấy danh sách dữ liệu đã giải mã từ lần detect gần nhất
     * @return Vector chứa các chuỗi string đã giải mã
     * @details Trả về chỉ phần data (string) của QRInfo, bỏ qua thông tin vị trí.
     */
    std::vector<std::string> getDecodedData() const; // Lấy dữ liệu giải mã

    /**
     * @brief Lấy lịch sử tất cả QR code đã quét
     * @return Vector chứa QRInfo của tất cả QR code từng được phát hiện
     * @details Lịch sử được lưu theo thứ tự thời gian (mới nhất ở cuối).
     *          Giới hạn bởi max_history_size_ để tránh tràn bộ nhớ.
     */
    std::vector<QRInfo> getHistory() const; // Lấy lịch sử quét

    /**
     * @brief Xóa toàn bộ lịch sử quét QR code
     * @details Giải phóng bộ nhớ của deque lịch sử. Thread-safe.
     */
    void clearHistory(); // Xóa lịch sử

    /**
     * @brief Thiết lập giới hạn số lượng mục trong lịch sử
     * @param max_size Số lượng tối đa mục lịch sử (mặc định 100)
     */
    void setMaxHistorySize(size_t max_size); // Đặt giới hạn lịch sử

    // ========================================================================
    // CÁC PHƯƠNG THỨC PHÂN TÍCH LỆNH DRONE (COMMAND PARSING METHODS)
    // ========================================================================

    /**
     * @brief Phân tích dữ liệu QR code thành lệnh điều khiển drone
     * @param data Chuỗi dữ liệu từ QR code (ví dụ: "WAYPOINT:21.0285,105.8542,100")
     * @return DroneCommand chứa loại lệnh và tham số đã phân tích
     * @details Định dạng lệnh: TYPE:param1,param2,...
     *          Các loại lệnh hỗ trợ:
     *          - WAYPOINT:lat,lon,alt      => params: {lat, lon, alt}
     *          - ACTION:takeoff|land|hover  => params: {action}
     *          - URL:https://...           => params: {url}
     *          - CONFIG:key1=val1,key2=val2 => params: {key1: val1, key2: val2}
     */
    DroneCommand parseCommand(const std::string& data); // Phân tích lệnh drone từ QR data

    // ========================================================================
    // CÁC PHƯƠNG THỨC TRUY VẤN BỔ SUNG (ADDITIONAL GETTERS)
    // ========================================================================

    /**
     * @brief Lấy danh sách QR code phát hiện trong lần detect gần nhất
     * @return Tham chiếu const đến vector QRInfo
     */
    const std::vector<QRInfo>& getDetectedQRs() const; // Lấy QR code đã phát hiện

    /**
     * @brief Lấy số lượng QR code phát hiện trong lần detect gần nhất
     * @return Số lượng QR code
     */
    size_t getQRCount() const; // Lấy số lượng QR code

    /**
     * @brief Lấy số lượng mục trong lịch sử
     * @return Số lượng mục lịch sử
     */
    size_t getHistorySize() const; // Lấy kích thước lịch sử

private: // Phần private - nội bộ lớp

    // ========================================================================
    // BIẾN THÀNH VIÊN PRIVATE
    // ========================================================================

    cv::QRCodeDetector qr_detector_;   // Đối tượng QRCodeDetector của OpenCV (xử lý phát hiện + giải mã nội bộ)

    std::vector<QRInfo> detected_qrs_; // Vector lưu QR code phát hiện trong lần detect() gần nhất

    std::deque<QRInfo> history_;        // Deque lưu lịch sử tất cả QR code đã quét (FIFO khi đầy)
                                        // Dùng deque thay vì vector để xóa phần tử đầu hiệu quả O(1)

    size_t max_history_size_;           // Giới hạn tối đa số mục trong lịch sử (mặc định 100)

    mutable std::mutex history_mutex_;  // Mutex bảo vệ truy cập lịch sử trong môi trường đa luồng
                                        // Khai báo mutable để có thể lock trong phương thức const

    // ========================================================================
    // PHƯƠNG THỨC PRIVATE HỖ TRỢ
    // ========================================================================

    /**
     * @brief Thêm QRInfo vào lịch sử, tự động xóa mục cũ nếu đầy
     * @param info QRInfo cần thêm vào lịch sử
     * @details Thread-safe: sử dụng lock_guard để bảo vệ truy cập.
     *          Nếu lịch sử đạt giới hạn max_history_size_, xóa mục cũ nhất (đầu deque).
     */
    void addToHistory(const QRInfo& info); // Thêm vào lịch sử

    /**
     * @brief Tính bounding box từ tọa độ 4 góc QR code
     * @param points Vector chứa tọa độ 4 góc
     * @return cv::Rect bounding box bao quanh QR code
     */
    cv::Rect computeBoundingBox(const std::vector<cv::Point2f>& points); // Tính bounding box

    /**
     * @brief Phân tích lệnh WAYPOINT: lat,lon,alt
     * @param params_str Chuỗi tham số sau dấu ':'
     * @param cmd Đối tượng DroneCommand sẽ được cập nhật
     */
    void parseWaypoint(const std::string& params_str, DroneCommand& cmd); // Parse WAYPOINT

    /**
     * @brief Phân tích lệnh ACTION: takeoff/land/hover/return
     * @param params_str Chuỗi tham số sau dấu ':'
     * @param cmd Đối tượng DroneCommand sẽ được cập nhật
     */
    void parseAction(const std::string& params_str, DroneCommand& cmd); // Parse ACTION

    /**
     * @brief Phân tích lệnh CONFIG: key1=val1,key2=val2
     * @param params_str Chuỗi tham số sau dấu ':'
     * @param cmd Đối tượng DroneCommand sẽ được cập nhật
     */
    void parseConfig(const std::string& params_str, DroneCommand& cmd); // Parse CONFIG
};

} // namespace drone_vision - Kết thúc namespace drone_vision
