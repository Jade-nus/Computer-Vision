/**
 * @file qr_detector.cpp
 * @brief Triển khai đầy đủ lớp QRDetector cho hệ thống DroneVisionPro
 * @details File nguồn chứa toàn bộ logic xử lý QR code:
 *          - Phát hiện và giải mã QR code đơn lẻ (detect)
 *          - Phát hiện nhiều QR code đồng thời (detectMulti)
 *          - Vẽ kết quả trực quan (drawDetections)
 *          - Quản lý lịch sử quét với timestamp và giới hạn kích thước
 *          - Phân tích dữ liệu QR thành lệnh drone (parseCommand)
 *          - Hỗ trợ lệnh: WAYPOINT, ACTION, URL, CONFIG
 * @author Tran Ngoc Bao - 24021238
 * @date 2026-07-13
 */

// ============================================================================
// PHẦN INCLUDE CÁC THƯ VIỆN CẦN THIẾT
// ============================================================================

#include "qr_detector.hpp"             // Include file header tương ứng chứa khai báo lớp QRDetector

#include <iostream>                    // Thư viện xuất/nhập chuẩn (std::cout, std::cerr) cho thông báo debug
#include <sstream>                     // Thư viện stringstream để phân tích chuỗi (parse command)
#include <algorithm>                   // Thư viện thuật toán STL (transform, min, max, ...)
#include <iomanip>                     // Thư viện format xuất (put_time, setw) để format timestamp
#include <ctime>                       // Thư viện thời gian C (localtime, strftime) hỗ trợ format

namespace drone_vision { // Mở namespace drone_vision tương ứng với file header

// ============================================================================
// CONSTRUCTOR - KHỞI TẠO ĐỐI TƯỢNG QRDETECTOR
// ============================================================================

QRDetector::QRDetector()               // Constructor mặc định không tham số
    : max_history_size_(100)           // Khởi tạo giới hạn lịch sử = 100 mục (tránh tràn bộ nhớ khi chạy lâu)
{
    // qr_detector_ (cv::QRCodeDetector) được tự động khởi tạo bởi constructor mặc định của nó
    // OpenCV QRCodeDetector không cần tham số cấu hình đặc biệt

    // In thông báo khởi tạo thành công ra console
    std::cout << "[QRDetector] Da khoi tao thanh cong" << std::endl;                // Thông báo constructor hoàn tất
    std::cout << "[QRDetector] Gioi han lich su: " << max_history_size_              // In giới hạn lịch sử
              << " muc" << std::endl;                                                 // Đơn vị: mục (entries)
} // Kết thúc constructor

// ============================================================================
// PHƯƠNG THỨC PHÁT HIỆN ĐƠN - DETECT (MỘT QR CODE)
// ============================================================================

std::vector<QRInfo> QRDetector::detect(const cv::Mat& frame) { // Phát hiện 1 QR code trong khung hình
    // Xóa dữ liệu phát hiện cũ từ lần gọi trước
    detected_qrs_.clear(); // Xóa danh sách QRInfo cũ để chuẩn bị cho lần phát hiện mới

    // Kiểm tra khung hình đầu vào có hợp lệ không
    if (frame.empty()) { // Nếu ảnh rỗng (không có dữ liệu pixel)
        std::cerr << "[QRDetector] CANH BAO: Khung hinh dau vao rong!" << std::endl; // In cảnh báo
        return detected_qrs_; // Trả về vector rỗng
    } // Kết thúc kiểm tra ảnh rỗng

    // Khai báo biến lưu kết quả phát hiện
    std::vector<cv::Point2f> points;   // Vector chứa tọa độ 4 góc QR code trên ảnh
    std::string decoded_data;          // Chuỗi chứa dữ liệu đã giải mã từ QR code

    try { // Bắt đầu khối try để xử lý ngoại lệ
        // Tạo biến Mat cho ảnh đã xử lý nội bộ (straightened QR image)
        cv::Mat straight_qr; // Ảnh QR code đã được nắn thẳng (rectified), dùng nội bộ bởi OpenCV

        // Gọi detectAndDecode của OpenCV QRCodeDetector
        // Hàm này kết hợp 2 bước: phát hiện vị trí QR + giải mã nội dung
        // Trả về chuỗi đã giải mã, hoặc chuỗi rỗng nếu không tìm thấy/không giải mã được
        decoded_data = qr_detector_.detectAndDecode(
            frame,                 // Ảnh đầu vào (BGR hoặc grayscale, OpenCV tự xử lý)
            points,                // [OUTPUT] Tọa độ 4 góc QR code (nếu phát hiện được)
            straight_qr            // [OUTPUT] Ảnh QR code đã nắn thẳng (optional, dùng cho debug)
        ); // Kết thúc detectAndDecode

        // Kiểm tra xem có phát hiện và giải mã thành công không
        if (!decoded_data.empty() && !points.empty()) { // Nếu cả dữ liệu và tọa độ đều có
            QRInfo info;                                   // Tạo đối tượng QRInfo mới cho QR code tìm được
            info.data = decoded_data;                      // Gán dữ liệu đã giải mã vào trường data
            info.points = points;                          // Gán tọa độ 4 góc vào trường points
            info.bbox = computeBoundingBox(points);        // Tính bounding box từ tọa độ 4 góc
            info.timestamp = std::chrono::system_clock::now(); // Gán timestamp = thời điểm hiện tại

            detected_qrs_.push_back(info);                 // Thêm QRInfo vào danh sách phát hiện
            addToHistory(info);                            // Thêm vào lịch sử quét (thread-safe)

            // In thông tin QR code đã phát hiện ra console
            std::cout << "[QRDetector] Da phat hien QR code: \"" << decoded_data   // In dữ liệu
                      << "\"" << std::endl;                                          // Kết thúc dòng
        } // Kết thúc kiểm tra phát hiện thành công
        else { // Nếu không phát hiện hoặc không giải mã được
            // Không in thông báo ở đây để tránh spam console khi không có QR code
            // (hàm detect được gọi liên tục mỗi frame)
        } // Kết thúc else
    }
    catch (const cv::Exception& e) { // Bắt ngoại lệ OpenCV nếu phát hiện gặp lỗi
        std::cerr << "[QRDetector] LOI khi phat hien QR code: " << e.what() << std::endl; // In lỗi
    } // Kết thúc catch

    return detected_qrs_; // Trả về vector QRInfo (0 hoặc 1 phần tử)
} // Kết thúc detect

// ============================================================================
// PHƯƠNG THỨC PHÁT HIỆN NHIỀU - DETECT MULTI (NHIỀU QR CODE)
// ============================================================================

std::vector<QRInfo> QRDetector::detectMulti(const cv::Mat& frame) { // Phát hiện nhiều QR code cùng lúc
    // Xóa dữ liệu phát hiện cũ
    detected_qrs_.clear(); // Xóa danh sách QRInfo cũ

    // Kiểm tra khung hình đầu vào
    if (frame.empty()) { // Nếu ảnh rỗng
        std::cerr << "[QRDetector] CANH BAO: Khung hinh dau vao rong!" << std::endl; // In cảnh báo
        return detected_qrs_; // Trả về vector rỗng
    } // Kết thúc kiểm tra

    try { // Bắt đầu khối try

        // Khai báo biến lưu kết quả phát hiện nhiều QR code
        std::vector<std::string> decoded_infos;    // Vector chứa dữ liệu giải mã của mỗi QR code
        std::vector<cv::Point2f> all_points;       // Vector chứa tọa độ góc của tất cả QR code (nối tiếp, mỗi QR 4 điểm)
        std::vector<cv::Mat> straight_qr_codes;    // Vector chứa ảnh QR đã nắn thẳng (cho debug)

        // Gọi detectAndDecodeMulti để phát hiện nhiều QR code đồng thời
        // Hàm này là phiên bản mở rộng của detectAndDecode, hỗ trợ từ OpenCV 4.x
        // Trả về true nếu phát hiện ít nhất 1 QR code
        bool found = qr_detector_.detectAndDecodeMulti(
            frame,                 // Ảnh đầu vào
            decoded_infos,         // [OUTPUT] Vector chứa chuỗi giải mã của mỗi QR
            all_points,            // [OUTPUT] Vector tọa độ góc (4 điểm/QR, nối tiếp)
            straight_qr_codes     // [OUTPUT] Vector ảnh QR đã nắn thẳng
        ); // Kết thúc detectAndDecodeMulti

        // Kiểm tra kết quả phát hiện
        if (found && !decoded_infos.empty()) { // Nếu phát hiện thành công và có dữ liệu

            // Duyệt qua từng QR code đã phát hiện
            for (size_t i = 0; i < decoded_infos.size(); ++i) { // Lặp từ QR đầu tiên đến QR cuối
                // Bỏ qua QR code có dữ liệu rỗng (phát hiện nhưng không giải mã được)
                if (decoded_infos[i].empty()) { // Nếu dữ liệu rỗng
                    continue; // Bỏ qua QR này, chuyển sang QR tiếp theo
                } // Kết thúc kiểm tra dữ liệu rỗng

                QRInfo info;                                       // Tạo QRInfo mới cho QR code thứ i
                info.data = decoded_infos[i];                      // Gán dữ liệu giải mã

                // Trích xuất 4 góc của QR code thứ i từ mảng all_points
                // Mỗi QR code có 4 góc, nên góc của QR thứ i bắt đầu từ vị trí i*4
                size_t start_idx = i * 4;                          // Chỉ số bắt đầu = i * 4 (mỗi QR 4 điểm)

                // Kiểm tra chỉ số không vượt quá kích thước mảng
                if (start_idx + 4 <= all_points.size()) {          // Đảm bảo đủ 4 điểm cho QR thứ i
                    // Sao chép 4 góc từ all_points vào info.points
                    info.points.assign(
                        all_points.begin() + static_cast<long>(start_idx),     // Iterator đầu: vị trí i*4
                        all_points.begin() + static_cast<long>(start_idx + 4)  // Iterator cuối: vị trí i*4 + 4
                    ); // Kết thúc assign - sao chép 4 phần tử

                    info.bbox = computeBoundingBox(info.points);   // Tính bounding box từ 4 góc
                } // Kết thúc kiểm tra chỉ số

                info.timestamp = std::chrono::system_clock::now(); // Gán timestamp = thời điểm hiện tại

                detected_qrs_.push_back(info);                     // Thêm vào danh sách phát hiện
                addToHistory(info);                                // Thêm vào lịch sử (thread-safe)

                // In thông tin QR code đã phát hiện
                std::cout << "[QRDetector] Multi - QR #" << i      // In chỉ số QR
                          << ": \"" << decoded_infos[i]             // In dữ liệu
                          << "\"" << std::endl;                     // Kết thúc dòng
            } // Kết thúc vòng lặp duyệt QR code

            // In tổng số QR code phát hiện được
            std::cout << "[QRDetector] Da phat hien " << detected_qrs_.size()   // Số lượng QR code
                      << " QR code(s)" << std::endl;                              // Kết thúc thông báo
        } // Kết thúc kiểm tra found

    }
    catch (const cv::Exception& e) { // Bắt ngoại lệ OpenCV
        // detectAndDecodeMulti có thể không khả dụng trên OpenCV cũ hơn
        std::cerr << "[QRDetector] LOI detectMulti (co the API khong kha dung): "  // Thông báo lỗi
                  << e.what() << std::endl;                                          // Chi tiết lỗi

        // Fallback: thử dùng detect() đơn lẻ thay thế
        std::cout << "[QRDetector] Fallback sang detect() don le" << std::endl;      // Thông báo fallback
        return detect(frame); // Gọi lại hàm detect đơn lẻ (ít nhất tìm được 1 QR)
    } // Kết thúc catch

    return detected_qrs_; // Trả về vector QRInfo chứa tất cả QR code phát hiện được
} // Kết thúc detectMulti

// ============================================================================
// PHƯƠNG THỨC VẼ KẾT QUẢ - DRAW DETECTIONS
// ============================================================================

void QRDetector::drawDetections(cv::Mat& frame) { // Vẽ kết quả phát hiện QR code lên khung hình
    // Kiểm tra khung hình có hợp lệ không
    if (frame.empty()) { // Nếu ảnh rỗng
        std::cerr << "[QRDetector] CANH BAO: Khung hinh rong, khong the ve!" << std::endl; // Cảnh báo
        return; // Thoát hàm
    } // Kết thúc kiểm tra

    // Kiểm tra có QR code nào để vẽ không
    if (detected_qrs_.empty()) { // Nếu không có QR code nào
        return; // Thoát hàm - không có gì để vẽ
    } // Kết thúc kiểm tra

    // Duyệt qua từng QR code đã phát hiện để vẽ
    for (size_t i = 0; i < detected_qrs_.size(); ++i) { // Lặp qua tất cả QR code
        const auto& qr = detected_qrs_[i]; // Tham chiếu const đến QR code thứ i

        // ====================================================================
        // BƯỚC 1: VẼ ĐƯỜNG NỐI 4 GÓC CỦA QR CODE
        // ====================================================================

        if (qr.points.size() >= 4) { // Kiểm tra có đủ 4 góc không
            // Vẽ 4 cạnh của QR code bằng cách nối 4 góc liên tiếp
            for (int j = 0; j < 4; ++j) { // Lặp 4 lần cho 4 cạnh
                // Vẽ đường thẳng từ góc j đến góc tiếp theo (j+1, quay vòng bằng %)
                cv::line(
                    frame,                                         // Ảnh đầu ra
                    cv::Point(                                     // Điểm bắt đầu: góc j
                        static_cast<int>(qr.points[j].x),         // Tọa độ x góc j (chuyển sang int)
                        static_cast<int>(qr.points[j].y)          // Tọa độ y góc j
                    ),
                    cv::Point(                                     // Điểm kết thúc: góc (j+1)%4
                        static_cast<int>(qr.points[(j + 1) % 4].x), // Tọa độ x góc tiếp theo
                        static_cast<int>(qr.points[(j + 1) % 4].y)  // Tọa độ y góc tiếp theo
                    ),
                    cv::Scalar(0, 255, 0),                         // Màu xanh lá (BGR: 0,255,0 = Green)
                    3                                              // Độ dày đường = 3 pixel
                ); // Kết thúc vẽ đường

                // Vẽ điểm tròn tại mỗi góc để đánh dấu rõ ràng
                cv::circle(
                    frame,                                         // Ảnh đầu ra
                    cv::Point(                                     // Tâm hình tròn = tọa độ góc j
                        static_cast<int>(qr.points[j].x),         // Tọa độ x
                        static_cast<int>(qr.points[j].y)          // Tọa độ y
                    ),
                    5,                                             // Bán kính 5 pixel
                    cv::Scalar(255, 0, 0),                         // Màu xanh dương (BGR: 255,0,0 = Blue)
                    -1                                             // Độ dày -1 = tô đặc (filled)
                ); // Kết thúc vẽ điểm
            } // Kết thúc vòng lặp 4 cạnh
        } // Kết thúc kiểm tra đủ góc

        // ====================================================================
        // BƯỚC 2: VẼ BOUNDING BOX
        // ====================================================================

        if (qr.bbox.width > 0 && qr.bbox.height > 0) { // Nếu bounding box hợp lệ (kích thước > 0)
            // Vẽ hình chữ nhật bao quanh QR code
            cv::rectangle(
                frame,                             // Ảnh đầu ra
                qr.bbox,                           // Hình chữ nhật bounding box
                cv::Scalar(0, 255, 255),           // Màu vàng (BGR: 0,255,255 = Yellow)
                2                                  // Độ dày đường viền = 2 pixel
            ); // Kết thúc vẽ rectangle
        } // Kết thúc kiểm tra bounding box

        // ====================================================================
        // BƯỚC 3: VẼ DỮ LIỆU ĐÃ GIẢI MÃ (TEXT) PHÍA TRÊN BOUNDING BOX
        // ====================================================================

        if (!qr.data.empty()) { // Nếu dữ liệu không rỗng

            // Cắt chuỗi hiển thị nếu quá dài (tối đa 50 ký tự + "...")
            std::string display_text = qr.data;                    // Sao chép dữ liệu gốc

            if (display_text.length() > 50) { // Nếu chuỗi dài hơn 50 ký tự
                display_text = display_text.substr(0, 47) + "..."; // Cắt và thêm "..." (ellipsis)
            } // Kết thúc cắt chuỗi

            // Tính vị trí vẽ text: phía trên bounding box, lệch lên 10 pixel
            cv::Point text_pos(
                qr.bbox.x,                        // X = cạnh trái bounding box
                qr.bbox.y - 10                     // Y = cạnh trên bounding box trừ 10 pixel (phía trên)
            ); // Kết thúc tạo vị trí

            // Đảm bảo text không bị vượt ra ngoài ảnh (y >= 15 để có chỗ vẽ)
            if (text_pos.y < 15) {                 // Nếu vị trí y quá gần cạnh trên ảnh
                text_pos.y = 15;                   // Đặt y = 15 pixel (cách cạnh trên 15 pixel)
            } // Kết thúc kiểm tra vị trí

            // Vẽ nền đen (bóng) cho text để tăng khả năng đọc
            cv::putText(
                frame,                             // Ảnh đầu ra
                display_text,                      // Chuỗi hiển thị
                text_pos + cv::Point(1, 1),        // Vị trí lệch 1 pixel (tạo bóng)
                cv::FONT_HERSHEY_SIMPLEX,          // Font chữ đơn giản, dễ đọc
                0.5,                               // Cỡ chữ = 0.5 (vừa phải)
                cv::Scalar(0, 0, 0),               // Màu đen cho bóng
                2                                  // Độ dày nét = 2 pixel
            ); // Kết thúc vẽ bóng

            // Vẽ text chính màu trắng (nổi bật trên nền đen/bóng)
            cv::putText(
                frame,                             // Ảnh đầu ra
                display_text,                      // Chuỗi hiển thị
                text_pos,                          // Vị trí text chính
                cv::FONT_HERSHEY_SIMPLEX,          // Font chữ
                0.5,                               // Cỡ chữ
                cv::Scalar(255, 255, 255),         // Màu trắng (BGR: 255,255,255 = White)
                1                                  // Độ dày nét = 1 pixel (nhẹ hơn bóng)
            ); // Kết thúc vẽ text chính
        } // Kết thúc kiểm tra dữ liệu không rỗng
    } // Kết thúc vòng lặp duyệt QR code

    // ========================================================================
    // VẼ THỐNG KÊ TỔNG SỐ QR CODE Ở GÓC TRÊN-TRÁI ẢNH
    // ========================================================================

    // Tạo chuỗi thống kê
    std::string count_text = "QR Codes: " + std::to_string(detected_qrs_.size()); // "QR Codes: N"
    std::string history_text = "History: " + std::to_string(history_.size());       // "History: M"

    // Vẽ số lượng QR code hiện tại
    cv::putText(
        frame,                                 // Ảnh đầu ra
        count_text,                            // Chuỗi "QR Codes: N"
        cv::Point(10, 60),                     // Vị trí góc trên-trái (10, 60) - dưới thống kê marker nếu có
        cv::FONT_HERSHEY_SIMPLEX,              // Font chữ
        0.7,                                   // Cỡ chữ
        cv::Scalar(255, 200, 0),               // Màu xanh dương nhạt (BGR)
        2                                      // Độ dày nét
    ); // Kết thúc vẽ text thống kê QR

    // Vẽ kích thước lịch sử
    cv::putText(
        frame,                                 // Ảnh đầu ra
        history_text,                          // Chuỗi "History: M"
        cv::Point(10, 85),                     // Vị trí phía dưới thống kê QR
        cv::FONT_HERSHEY_SIMPLEX,              // Font chữ
        0.5,                                   // Cỡ chữ nhỏ hơn
        cv::Scalar(200, 200, 200),             // Màu xám nhạt
        1                                      // Độ dày nét
    ); // Kết thúc vẽ text lịch sử
} // Kết thúc drawDetections

// ============================================================================
// PHƯƠNG THỨC LẤY DỮ LIỆU ĐÃ GIẢI MÃ - GET DECODED DATA
// ============================================================================

std::vector<std::string> QRDetector::getDecodedData() const { // Lấy danh sách dữ liệu giải mã
    std::vector<std::string> result; // Tạo vector kết quả

    // Trích xuất phần data từ mỗi QRInfo
    for (const auto& qr : detected_qrs_) { // Duyệt qua tất cả QR code đã phát hiện
        if (!qr.data.empty()) { // Chỉ thêm nếu dữ liệu không rỗng
            result.push_back(qr.data); // Thêm chuỗi data vào vector kết quả
        } // Kết thúc kiểm tra
    } // Kết thúc vòng lặp

    return result; // Trả về vector chứa các chuỗi dữ liệu đã giải mã
} // Kết thúc getDecodedData

// ============================================================================
// PHƯƠNG THỨC LẤY LỊCH SỬ - GET HISTORY
// ============================================================================

std::vector<QRInfo> QRDetector::getHistory() const { // Lấy toàn bộ lịch sử quét QR code
    std::lock_guard<std::mutex> lock(history_mutex_); // Lock mutex để đảm bảo thread-safe khi đọc

    // Chuyển deque thành vector để trả về (interface đồng nhất)
    std::vector<QRInfo> result(history_.begin(), history_.end()); // Sao chép từ deque sang vector

    return result; // Trả về vector chứa toàn bộ lịch sử
} // Kết thúc getHistory - lock_guard tự động unlock mutex khi ra khỏi scope

// ============================================================================
// PHƯƠNG THỨC XÓA LỊCH SỬ - CLEAR HISTORY
// ============================================================================

void QRDetector::clearHistory() { // Xóa toàn bộ lịch sử quét
    std::lock_guard<std::mutex> lock(history_mutex_); // Lock mutex để đảm bảo thread-safe

    history_.clear(); // Xóa tất cả phần tử trong deque lịch sử

    std::cout << "[QRDetector] Da xoa toan bo lich su quet" << std::endl; // Thông báo xóa thành công
} // Kết thúc clearHistory - lock_guard tự động unlock

// ============================================================================
// PHƯƠNG THỨC THIẾT LẬP GIỚI HẠN LỊCH SỬ - SET MAX HISTORY SIZE
// ============================================================================

void QRDetector::setMaxHistorySize(size_t max_size) { // Đặt giới hạn số mục trong lịch sử
    std::lock_guard<std::mutex> lock(history_mutex_); // Lock mutex để đảm bảo thread-safe

    max_history_size_ = max_size; // Cập nhật giới hạn mới

    // Nếu lịch sử hiện tại vượt quá giới hạn mới, xóa bớt phần tử cũ
    while (history_.size() > max_history_size_) { // Lặp cho đến khi kích thước <= giới hạn
        history_.pop_front(); // Xóa phần tử cũ nhất (đầu deque) - FIFO
    } // Kết thúc vòng lặp xóa bớt

    std::cout << "[QRDetector] Da thiet lap gioi han lich su: " << max_size  // Thông báo
              << " muc" << std::endl;                                          // Kết thúc dòng
} // Kết thúc setMaxHistorySize

// ============================================================================
// PHƯƠNG THỨC PHÂN TÍCH LỆNH DRONE - PARSE COMMAND
// ============================================================================

DroneCommand QRDetector::parseCommand(const std::string& data) { // Phân tích QR data thành lệnh drone
    DroneCommand cmd;          // Tạo đối tượng DroneCommand (mặc định: type="UNKNOWN", valid=false)
    cmd.raw_data = data;       // Lưu dữ liệu gốc để debug và tham chiếu

    // Kiểm tra dữ liệu có rỗng không
    if (data.empty()) { // Nếu chuỗi rỗng
        std::cerr << "[QRDetector] CANH BAO: Du lieu QR rong, khong the parse!" << std::endl; // Cảnh báo
        return cmd; // Trả về lệnh UNKNOWN
    } // Kết thúc kiểm tra rỗng

    // Tìm vị trí dấu ':' phân tách TYPE và params
    // Định dạng: TYPE:param1,param2,...
    size_t colon_pos = data.find(':'); // Tìm vị trí đầu tiên của dấu ':'

    // Kiểm tra có dấu ':' không
    if (colon_pos == std::string::npos) { // Nếu không tìm thấy dấu ':'
        // Dữ liệu không theo định dạng lệnh drone, coi là text thuần
        std::cout << "[QRDetector] Du lieu QR khong phai lenh drone: \"" << data   // Thông báo
                  << "\"" << std::endl;                                              // Kết thúc
        cmd.type = "TEXT";                 // Đặt loại = TEXT (dữ liệu văn bản thuần)
        cmd.params["content"] = data;      // Lưu toàn bộ dữ liệu vào params với key "content"
        cmd.valid = true;                  // Đánh dấu hợp lệ (vẫn parse được, chỉ không phải lệnh drone)
        return cmd; // Trả về lệnh TEXT
    } // Kết thúc kiểm tra dấu ':'

    // Tách TYPE (phần trước dấu ':') và params_str (phần sau dấu ':')
    std::string type_str = data.substr(0, colon_pos);          // Trích xuất TYPE: từ đầu đến trước ':'
    std::string params_str = data.substr(colon_pos + 1);       // Trích xuất params: từ sau ':' đến hết

    // Chuyển TYPE sang chữ hoa để so sánh không phân biệt hoa-thường
    std::transform(
        type_str.begin(),              // Iterator đầu chuỗi
        type_str.end(),                // Iterator cuối chuỗi
        type_str.begin(),              // Iterator đầu ra (ghi đè vào chính chuỗi)
        ::toupper                      // Hàm chuyển từng ký tự sang chữ hoa
    ); // Kết thúc transform - type_str giờ là chữ hoa

    // Gán loại lệnh
    cmd.type = type_str; // Gán TYPE đã chuẩn hóa (chữ hoa) vào DroneCommand

    // In thông tin lệnh đang phân tích
    std::cout << "[QRDetector] Dang parse lenh - Type: " << type_str       // In loại lệnh
              << " | Params: " << params_str << std::endl;                   // In tham số

    // ========================================================================
    // PHÂN TÍCH THEO TỪNG LOẠI LỆNH
    // ========================================================================

    if (type_str == "WAYPOINT") { // Nếu lệnh WAYPOINT: lat,lon,alt
        parseWaypoint(params_str, cmd); // Gọi hàm phân tích WAYPOINT chuyên biệt
    }
    else if (type_str == "ACTION") { // Nếu lệnh ACTION: takeoff/land/hover/return
        parseAction(params_str, cmd); // Gọi hàm phân tích ACTION chuyên biệt
    }
    else if (type_str == "URL") { // Nếu lệnh URL: https://...
        cmd.params["url"] = params_str; // Lưu toàn bộ URL vào params
        cmd.valid = true;               // URL luôn hợp lệ (không cần validate format)

        std::cout << "[QRDetector] Lenh URL: " << params_str << std::endl; // In URL
    }
    else if (type_str == "CONFIG") { // Nếu lệnh CONFIG: key1=val1,key2=val2
        parseConfig(params_str, cmd); // Gọi hàm phân tích CONFIG chuyên biệt
    }
    else { // Lệnh không nhận dạng được
        std::cout << "[QRDetector] Lenh khong nhan dang: " << type_str << std::endl; // Thông báo
        cmd.type = "UNKNOWN";          // Đặt lại type = UNKNOWN
        cmd.params["raw"] = data;      // Lưu dữ liệu gốc vào params
        cmd.valid = false;             // Đánh dấu không hợp lệ
    } // Kết thúc chuỗi if-else phân loại lệnh

    return cmd; // Trả về DroneCommand đã phân tích
} // Kết thúc parseCommand

// ============================================================================
// PHƯƠNG THỨC TRUY VẤN - GETTERS
// ============================================================================

const std::vector<QRInfo>& QRDetector::getDetectedQRs() const { // Lấy danh sách QR đã phát hiện
    return detected_qrs_; // Trả về tham chiếu const đến vector QRInfo
} // Kết thúc getDetectedQRs

size_t QRDetector::getQRCount() const { // Lấy số lượng QR code phát hiện được
    return detected_qrs_.size(); // Trả về kích thước vector
} // Kết thúc getQRCount

size_t QRDetector::getHistorySize() const { // Lấy kích thước lịch sử
    std::lock_guard<std::mutex> lock(history_mutex_); // Lock mutex để thread-safe
    return history_.size(); // Trả về số phần tử trong deque lịch sử
} // Kết thúc getHistorySize

// ============================================================================
// PHƯƠNG THỨC PRIVATE - THÊM VÀO LỊCH SỬ
// ============================================================================

void QRDetector::addToHistory(const QRInfo& info) { // Thêm QRInfo vào lịch sử
    std::lock_guard<std::mutex> lock(history_mutex_); // Lock mutex để đảm bảo thread-safe khi ghi

    // Thêm mục mới vào cuối deque
    history_.push_back(info); // Thêm QRInfo vào cuối hàng đợi

    // Kiểm tra và xóa mục cũ nếu vượt giới hạn
    while (history_.size() > max_history_size_) { // Nếu kích thước > giới hạn
        history_.pop_front(); // Xóa mục cũ nhất ở đầu deque (FIFO - First In First Out)
    } // Kết thúc vòng lặp kiểm tra giới hạn
} // Kết thúc addToHistory - lock_guard tự động unlock mutex

// ============================================================================
// PHƯƠNG THỨC PRIVATE - TÍNH BOUNDING BOX
// ============================================================================

cv::Rect QRDetector::computeBoundingBox(const std::vector<cv::Point2f>& points) { // Tính bbox từ 4 góc
    // Kiểm tra đầu vào có hợp lệ không
    if (points.empty()) { // Nếu vector rỗng
        return cv::Rect(0, 0, 0, 0); // Trả về rect rỗng (vị trí 0, kích thước 0)
    } // Kết thúc kiểm tra

    // Khởi tạo giá trị min/max với giá trị đầu tiên
    float min_x = points[0].x; // X nhỏ nhất, khởi tạo = x điểm đầu
    float max_x = points[0].x; // X lớn nhất, khởi tạo = x điểm đầu
    float min_y = points[0].y; // Y nhỏ nhất, khởi tạo = y điểm đầu
    float max_y = points[0].y; // Y lớn nhất, khởi tạo = y điểm đầu

    // Duyệt qua tất cả điểm để tìm min/max
    for (size_t i = 1; i < points.size(); ++i) { // Bắt đầu từ điểm thứ 2 (i=1)
        min_x = std::min(min_x, points[i].x); // Cập nhật x nhỏ nhất
        max_x = std::max(max_x, points[i].x); // Cập nhật x lớn nhất
        min_y = std::min(min_y, points[i].y); // Cập nhật y nhỏ nhất
        max_y = std::max(max_y, points[i].y); // Cập nhật y lớn nhất
    } // Kết thúc vòng lặp tìm min/max

    // Tạo và trả về Rect từ min/max
    // Rect(x, y, width, height): x,y = góc trên-trái, width = max_x - min_x, height = max_y - min_y
    return cv::Rect(
        static_cast<int>(min_x),                       // X góc trên-trái = min_x (chuyển sang int)
        static_cast<int>(min_y),                       // Y góc trên-trái = min_y
        static_cast<int>(max_x - min_x),               // Chiều rộng = max_x - min_x
        static_cast<int>(max_y - min_y)                // Chiều cao = max_y - min_y
    ); // Kết thúc tạo Rect
} // Kết thúc computeBoundingBox

// ============================================================================
// PHƯƠNG THỨC PRIVATE - PHÂN TÍCH LỆNH WAYPOINT
// ============================================================================

void QRDetector::parseWaypoint(const std::string& params_str, DroneCommand& cmd) { // Parse WAYPOINT
    // Định dạng WAYPOINT: lat,lon,alt
    // Ví dụ: "WAYPOINT:21.0285,105.8542,100.5"

    std::istringstream iss(params_str); // Tạo string stream từ chuỗi tham số để phân tích
    std::string token;                  // Biến tạm lưu từng token (phần tách bởi dấu ',')

    std::vector<std::string> tokens;    // Vector lưu tất cả token đã tách

    // Tách chuỗi bằng dấu ',' (delimiter)
    while (std::getline(iss, token, ',')) { // Đọc từng token từ stream, tách bởi ','
        // Xóa khoảng trắng đầu cuối của token (trim)
        size_t start = token.find_first_not_of(" \t"); // Tìm ký tự đầu tiên không phải khoảng trắng
        size_t end = token.find_last_not_of(" \t");     // Tìm ký tự cuối cùng không phải khoảng trắng

        if (start != std::string::npos) { // Nếu tìm thấy ký tự không phải khoảng trắng
            tokens.push_back(token.substr(start, end - start + 1)); // Thêm token đã trim
        } // Kết thúc trim
    } // Kết thúc tách chuỗi

    // WAYPOINT cần đúng 3 tham số: lat, lon, alt
    if (tokens.size() >= 3) { // Nếu có đủ 3 tham số trở lên
        cmd.params["lat"] = tokens[0]; // Tham số 1: latitude (vĩ độ)
        cmd.params["lon"] = tokens[1]; // Tham số 2: longitude (kinh độ)
        cmd.params["alt"] = tokens[2]; // Tham số 3: altitude (độ cao, mét)
        cmd.valid = true;              // Đánh dấu lệnh hợp lệ

        // In thông tin waypoint đã parse
        std::cout << "[QRDetector] WAYPOINT - Lat: " << tokens[0]  // In vĩ độ
                  << ", Lon: " << tokens[1]                          // In kinh độ
                  << ", Alt: " << tokens[2] << std::endl;            // In độ cao
    }
    else { // Nếu thiếu tham số
        std::cerr << "[QRDetector] LOI: WAYPOINT can 3 tham so (lat,lon,alt), " // Thông báo lỗi
                  << "nhan duoc " << tokens.size() << std::endl;                  // In số tham số nhận được
        cmd.valid = false; // Đánh dấu lệnh không hợp lệ
    } // Kết thúc kiểm tra số tham số
} // Kết thúc parseWaypoint

// ============================================================================
// PHƯƠNG THỨC PRIVATE - PHÂN TÍCH LỆNH ACTION
// ============================================================================

void QRDetector::parseAction(const std::string& params_str, DroneCommand& cmd) { // Parse ACTION
    // Định dạng ACTION: action_name
    // Ví dụ: "ACTION:takeoff", "ACTION:land", "ACTION:hover", "ACTION:return"

    // Chuẩn hóa tên action sang chữ thường để so sánh không phân biệt hoa/thường
    std::string action = params_str;   // Sao chép chuỗi tham số

    // Xóa khoảng trắng đầu cuối
    size_t start = action.find_first_not_of(" \t"); // Tìm ký tự đầu tiên không phải khoảng trắng
    size_t end = action.find_last_not_of(" \t");     // Tìm ký tự cuối cùng không phải khoảng trắng

    if (start != std::string::npos) { // Nếu tìm thấy ký tự không phải khoảng trắng
        action = action.substr(start, end - start + 1); // Trim chuỗi
    } // Kết thúc trim

    // Chuyển sang chữ thường
    std::transform(
        action.begin(), action.end(), // Range đầu vào
        action.begin(),               // Iterator đầu ra
        ::tolower                     // Hàm chuyển sang chữ thường
    ); // Kết thúc transform

    // Kiểm tra action có hợp lệ không (nằm trong danh sách cho phép)
    // Danh sách action hợp lệ cho drone
    const std::vector<std::string> valid_actions = { // Mảng const chứa các action được phép
        "takeoff",     // Cất cánh - drone bay lên độ cao mặc định
        "land",        // Hạ cánh - drone từ từ hạ xuống đất
        "hover",       // Bay treo - drone giữ nguyên vị trí hiện tại
        "return",      // Quay về - drone bay về điểm xuất phát (home)
        "emergency"    // Khẩn cấp - drone dừng động cơ ngay lập tức
    }; // Kết thúc danh sách action hợp lệ

    // Kiểm tra action có trong danh sách không
    bool is_valid_action = std::find(
        valid_actions.begin(),         // Iterator đầu danh sách
        valid_actions.end(),           // Iterator cuối danh sách
        action                         // Giá trị cần tìm
    ) != valid_actions.end(); // Trả về true nếu tìm thấy (iterator != end)

    // Gán kết quả
    cmd.params["action"] = action;     // Lưu tên action vào params
    cmd.valid = is_valid_action;       // Đánh dấu hợp lệ nếu action nằm trong danh sách

    // In thông tin action
    if (is_valid_action) { // Nếu action hợp lệ
        std::cout << "[QRDetector] ACTION hop le: " << action << std::endl; // Thông báo hợp lệ
    }
    else { // Nếu action không hợp lệ
        std::cerr << "[QRDetector] CANH BAO: ACTION khong hop le: " << action << std::endl; // Cảnh báo
    } // Kết thúc kiểm tra hợp lệ
} // Kết thúc parseAction

// ============================================================================
// PHƯƠNG THỨC PRIVATE - PHÂN TÍCH LỆNH CONFIG
// ============================================================================

void QRDetector::parseConfig(const std::string& params_str, DroneCommand& cmd) { // Parse CONFIG
    // Định dạng CONFIG: key1=val1,key2=val2,...
    // Ví dụ: "CONFIG:speed=5.0,altitude=100,mode=manual"

    std::istringstream iss(params_str); // Tạo string stream để phân tích
    std::string pair_str;               // Biến tạm lưu từng cặp key=value

    bool has_valid_pair = false;        // Cờ đánh dấu có ít nhất 1 cặp hợp lệ

    // Tách chuỗi bằng dấu ',' để lấy từng cặp key=value
    while (std::getline(iss, pair_str, ',')) { // Đọc từng cặp, tách bởi ','
        // Tìm vị trí dấu '=' phân tách key và value
        size_t eq_pos = pair_str.find('='); // Tìm vị trí dấu '='

        if (eq_pos != std::string::npos) { // Nếu tìm thấy dấu '='
            // Tách key (trước '=') và value (sau '=')
            std::string key = pair_str.substr(0, eq_pos);      // Key = phần trước '='
            std::string value = pair_str.substr(eq_pos + 1);   // Value = phần sau '='

            // Trim khoảng trắng cho key
            size_t k_start = key.find_first_not_of(" \t");     // Tìm ký tự đầu không phải khoảng trắng
            size_t k_end = key.find_last_not_of(" \t");        // Tìm ký tự cuối không phải khoảng trắng

            if (k_start != std::string::npos) { // Nếu key không phải toàn khoảng trắng
                key = key.substr(k_start, k_end - k_start + 1); // Trim key
            } // Kết thúc trim key

            // Trim khoảng trắng cho value
            size_t v_start = value.find_first_not_of(" \t");   // Tìm ký tự đầu
            size_t v_end = value.find_last_not_of(" \t");      // Tìm ký tự cuối

            if (v_start != std::string::npos) { // Nếu value không phải toàn khoảng trắng
                value = value.substr(v_start, v_end - v_start + 1); // Trim value
            } // Kết thúc trim value

            // Lưu cặp key-value vào params
            if (!key.empty()) { // Kiểm tra key không rỗng sau khi trim
                cmd.params[key] = value;       // Thêm vào map params
                has_valid_pair = true;         // Đánh dấu có cặp hợp lệ

                // In cặp key-value đã parse
                std::cout << "[QRDetector] CONFIG - " << key << " = " << value << std::endl; // In cặp
            } // Kết thúc kiểm tra key
        }
        else { // Nếu không có dấu '=' => cặp không hợp lệ
            std::cerr << "[QRDetector] CANH BAO: Cap config khong hop le: \""  // Cảnh báo
                      << pair_str << "\"" << std::endl;                          // In cặp lỗi
        } // Kết thúc kiểm tra dấu '='
    } // Kết thúc vòng lặp tách cặp

    cmd.valid = has_valid_pair; // Lệnh hợp lệ nếu có ít nhất 1 cặp key=value hợp lệ
} // Kết thúc parseConfig

} // namespace drone_vision - Kết thúc namespace drone_vision
