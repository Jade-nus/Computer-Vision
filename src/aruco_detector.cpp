/**
 * @file aruco_detector.cpp
 * @brief Triển khai đầy đủ lớp ArucoDetector cho hệ thống DroneVisionPro
 * @details File nguồn chứa toàn bộ logic xử lý ArUco marker:
 *          - Phát hiện marker trong khung hình camera
 *          - Ước lượng tư thế 3D sử dụng PnP
 *          - Tính khoảng cách Euclidean từ camera đến marker
 *          - Vẽ kết quả trực quan (viền, ID, trục 3D, khoảng cách)
 *          - Tạo hình ảnh marker để in ấn
 *          - Phát hiện và theo dõi marker bãi đáp
 * @author Tran Ngoc Bao - 24021238
 * @date 2026-07-13
 */

// ============================================================================
// PHẦN INCLUDE CÁC THƯ VIỆN CẦN THIẾT
// ============================================================================

#include "aruco_detector.hpp"          // Include file header tương ứng chứa khai báo lớp ArucoDetector

#include <iostream>                    // Thư viện xuất/nhập chuẩn (std::cout, std::cerr) để in thông báo debug
#include <sstream>                     // Thư viện stringstream để xây dựng chuỗi định dạng (format string)
#include <stdexcept>                   // Thư viện ngoại lệ chuẩn (runtime_error, invalid_argument) để xử lý lỗi
#include <algorithm>                   // Thư viện thuật toán STL (find_if, sort, ...) để tìm kiếm marker

namespace drone_vision { // Mở namespace drone_vision tương ứng với file header

// ============================================================================
// CONSTRUCTOR - KHỞI TẠO ĐỐI TƯỢNG ARUCODETECTOR
// ============================================================================

ArucoDetector::ArucoDetector(
    const cv::Mat& camera_matrix,      // Tham số: ma trận nội tại camera (3x3, chứa fx, fy, cx, cy)
    const cv::Mat& dist_coeffs         // Tham số: vector hệ số méo (5x1, chứa k1, k2, p1, p2, k3)
)
    : camera_matrix_(camera_matrix.clone()) // Sao chép sâu (deep copy) ma trận camera để tránh chia sẻ bộ nhớ với bên ngoài
    , dist_coeffs_(dist_coeffs.clone())     // Sao chép sâu vector hệ số méo để đảm bảo dữ liệu độc lập
    , landing_pad_id_(0)                    // Khởi tạo ID bãi đáp mặc định là 0 (marker ID=0 là bãi đáp)
    , pose_estimated_(false)                // Ban đầu chưa ước lượng pose nên đặt cờ = false
{
    // Khởi tạo dictionary ArUco mặc định là DICT_4X4_50
    // DICT_4X4_50 có 50 marker, mỗi marker 4x4 bit - phù hợp cho ứng dụng drone cơ bản
    dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50); // Lấy dictionary có sẵn từ OpenCV

    // Khởi tạo tham số detector với giá trị mặc định của OpenCV
    // Các tham số này kiểm soát quá trình phát hiện: ngưỡng adaptive, kích thước tối thiểu, ...
    parameters_ = cv::aruco::DetectorParameters::create(); // Tạo đối tượng DetectorParameters với giá trị mặc định

    // In thông báo khởi tạo thành công ra console để debug
    std::cout << "[ArucoDetector] Da khoi tao thanh cong voi dictionary DICT_4X4_50" << std::endl; // Thông báo tiếng Việt không dấu
    std::cout << "[ArucoDetector] Camera matrix size: " << camera_matrix_.size() << std::endl;     // In kích thước ma trận camera
    std::cout << "[ArucoDetector] Dist coeffs size: " << dist_coeffs_.size() << std::endl;         // In kích thước vector hệ số méo
} // Kết thúc constructor

// ============================================================================
// PHƯƠNG THỨC CẤU HÌNH - THIẾT LẬP DICTIONARY
// ============================================================================

void ArucoDetector::setDictionary(int dict_type) { // Phương thức thiết lập loại dictionary ArUco
    // Sử dụng try-catch để xử lý lỗi khi dict_type không hợp lệ
    try { // Bắt đầu khối try - thử thực hiện thao tác có thể gây lỗi
        // Gọi getPredefinedDictionary với kiểu dictionary mới
        // Hàm này trả về con trỏ Ptr<Dictionary> đến dictionary tương ứng
        dictionary_ = cv::aruco::getPredefinedDictionary(dict_type); // Lấy dictionary mới từ OpenCV theo dict_type

        // In thông báo đã thay đổi dictionary thành công
        std::cout << "[ArucoDetector] Da thay doi dictionary sang loai: " << dict_type << std::endl; // Thông báo thay đổi
    }
    catch (const cv::Exception& e) { // Bắt ngoại lệ OpenCV nếu dict_type không hợp lệ
        // In cảnh báo lỗi ra stderr (luồng lỗi chuẩn)
        std::cerr << "[ArucoDetector] LOI: Khong the thiet lap dictionary loai " // Thông báo lỗi phần 1
                  << dict_type << ": " << e.what() << std::endl;                 // Kèm theo chi tiết lỗi từ OpenCV
    } // Kết thúc catch
} // Kết thúc setDictionary

// ============================================================================
// PHƯƠNG THỨC CẤU HÌNH - THIẾT LẬP ID BÃI ĐÁP
// ============================================================================

void ArucoDetector::setLandingPadId(int id) { // Phương thức đặt ID marker bãi đáp cho drone
    landing_pad_id_ = id; // Gán ID mới cho biến thành viên landing_pad_id_

    // In thông báo đã thay đổi ID bãi đáp
    std::cout << "[ArucoDetector] Da thiet lap Landing Pad ID = " << id << std::endl; // Thông báo xác nhận
} // Kết thúc setLandingPadId

// ============================================================================
// PHƯƠNG THỨC CẤU HÌNH - CẬP NHẬT THÔNG SỐ CAMERA
// ============================================================================

void ArucoDetector::setCameraParameters(
    const cv::Mat& camera_matrix,      // Ma trận nội tại camera mới (3x3)
    const cv::Mat& dist_coeffs         // Vector hệ số méo mới
) { // Phương thức cập nhật thông số camera sau khi hiệu chuẩn
    camera_matrix_ = camera_matrix.clone(); // Sao chép sâu ma trận camera mới vào biến thành viên
    dist_coeffs_ = dist_coeffs.clone();     // Sao chép sâu vector hệ số méo mới vào biến thành viên

    // In thông báo cập nhật thành công
    std::cout << "[ArucoDetector] Da cap nhat thong so camera" << std::endl;                      // Thông báo chung
    std::cout << "[ArucoDetector] Camera matrix moi: " << camera_matrix_.size() << std::endl;     // In kích thước mới
    std::cout << "[ArucoDetector] Dist coeffs moi: " << dist_coeffs_.size() << std::endl;         // In kích thước mới
} // Kết thúc setCameraParameters

// ============================================================================
// PHƯƠNG THỨC PHÁT HIỆN CHÍNH - DETECT MARKERS
// ============================================================================

std::vector<MarkerInfo> ArucoDetector::detect(const cv::Mat& frame) { // Phương thức phát hiện ArUco marker trong khung hình
    // Xóa dữ liệu phát hiện cũ từ lần gọi trước để chuẩn bị cho lần phát hiện mới
    detected_markers_.clear();         // Xóa danh sách MarkerInfo cũ
    marker_corners_.clear();           // Xóa danh sách tọa độ góc cũ
    marker_ids_.clear();               // Xóa danh sách ID cũ
    rejected_candidates_.clear();      // Xóa danh sách ứng viên bị loại bỏ cũ
    pose_estimated_ = false;           // Reset cờ ước lượng pose về false cho chu kỳ mới

    // Kiểm tra khung hình đầu vào có hợp lệ không (không rỗng)
    if (frame.empty()) { // Nếu ảnh rỗng (không có dữ liệu pixel)
        // In cảnh báo ra stderr
        std::cerr << "[ArucoDetector] CANH BAO: Khung hinh dau vao rong!" << std::endl; // Thông báo ảnh rỗng
        return detected_markers_; // Trả về vector rỗng vì không có gì để phát hiện
    } // Kết thúc kiểm tra ảnh rỗng

    // Tạo biến lưu ảnh xám (grayscale) để tối ưu tốc độ phát hiện
    cv::Mat gray_frame; // Khai báo biến Mat để chứa ảnh grayscale

    // Kiểm tra số kênh màu của ảnh đầu vào
    if (frame.channels() == 3) { // Nếu ảnh có 3 kênh (BGR - ảnh màu)
        // Chuyển đổi từ BGR sang grayscale vì ArUco detector chỉ cần ảnh xám
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY); // Chuyển BGR -> Grayscale, giảm dữ liệu 3 lần
    }
    else if (frame.channels() == 4) { // Nếu ảnh có 4 kênh (BGRA - ảnh màu có alpha)
        // Chuyển đổi từ BGRA sang grayscale, bỏ kênh alpha (độ trong suốt)
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGRA2GRAY); // Chuyển BGRA -> Grayscale
    }
    else { // Trường hợp còn lại: ảnh đã là grayscale (1 kênh)
        gray_frame = frame; // Sử dụng trực tiếp ảnh đầu vào, không cần chuyển đổi (shallow copy)
    } // Kết thúc kiểm tra số kênh

    // ========================================================================
    // BƯỚC CHÍNH: PHÁT HIỆN ARUCO MARKERS SỬ DỤNG OPENCV API
    // ========================================================================

    try { // Bắt đầu khối try để xử lý ngoại lệ có thể xảy ra khi phát hiện
        // Gọi hàm detectMarkers của OpenCV ArUco module
        // Hàm này tìm tất cả ArUco marker trong ảnh dựa trên dictionary đã chọn
        // Output: marker_corners_ (tọa độ 4 góc), marker_ids_ (ID), rejected_candidates_ (ứng viên bị loại)
        cv::aruco::detectMarkers(
            gray_frame,                // Ảnh đầu vào (grayscale) để tìm marker
            dictionary_,               // Dictionary ArUco đang sử dụng (xác định bộ marker hợp lệ)
            marker_corners_,           // [OUTPUT] Vector 2D chứa tọa độ 4 góc của mỗi marker tìm được
            marker_ids_,               // [OUTPUT] Vector chứa ID tương ứng của mỗi marker
            parameters_,               // Tham số cấu hình detector (ngưỡng, kích thước tối thiểu, ...)
            rejected_candidates_       // [OUTPUT] Vector chứa các vùng giống marker nhưng bị loại bỏ
        ); // Kết thúc gọi detectMarkers
    }
    catch (const cv::Exception& e) { // Bắt ngoại lệ nếu detectMarkers gặp lỗi
        // In chi tiết lỗi ra stderr
        std::cerr << "[ArucoDetector] LOI khi phat hien marker: " << e.what() << std::endl; // Thông báo lỗi
        return detected_markers_; // Trả về vector rỗng nếu phát hiện thất bại
    } // Kết thúc catch

    // ========================================================================
    // XỬ LÝ KẾT QUẢ: TẠO MARKERINFO CHO MỖI MARKER PHÁT HIỆN ĐƯỢC
    // ========================================================================

    // Kiểm tra có marker nào được phát hiện không
    if (!marker_ids_.empty()) { // Nếu vector ID không rỗng (tìm thấy ít nhất 1 marker)
        // Duyệt qua tất cả marker đã phát hiện
        for (size_t i = 0; i < marker_ids_.size(); ++i) { // Lặp từ marker đầu tiên (i=0) đến marker cuối cùng
            MarkerInfo info;                    // Tạo đối tượng MarkerInfo mới cho marker thứ i
            info.id = marker_ids_[i];           // Gán ID của marker thứ i từ kết quả detectMarkers
            info.corners = marker_corners_[i];  // Gán tọa độ 4 góc của marker thứ i
            info.pose_valid = false;            // Đánh dấu pose chưa được ước lượng (sẽ cập nhật sau khi gọi estimatePose)
            info.distance = 0.0;                // Khoảng cách ban đầu = 0 (sẽ cập nhật sau khi gọi calculateDistance)

            detected_markers_.push_back(info);  // Thêm MarkerInfo vào cuối vector detected_markers_
        } // Kết thúc vòng lặp duyệt marker

        // In số lượng marker đã phát hiện để debug
        std::cout << "[ArucoDetector] Da phat hien " << detected_markers_.size() // In số lượng marker
                  << " ArUco marker(s)" << std::endl;                             // Kết thúc dòng thông báo
    } // Kết thúc kiểm tra marker_ids_ không rỗng

    return detected_markers_; // Trả về vector chứa thông tin tất cả marker đã phát hiện
} // Kết thúc detect

// ============================================================================
// PHƯƠNG THỨC ƯỚC LƯỢNG TƯ THẾ 3D - ESTIMATE POSE
// ============================================================================

void ArucoDetector::estimatePose(float marker_length) { // Ước lượng tư thế 3D của marker, cần kích thước thực
    // Kiểm tra xem có marker nào được phát hiện không
    if (detected_markers_.empty()) { // Nếu chưa phát hiện marker nào
        // In cảnh báo ra console
        std::cerr << "[ArucoDetector] CANH BAO: Chua phat hien marker nao, " // Thông báo phần 1
                  << "khong the uoc luong pose!" << std::endl;                 // Thông báo phần 2
        return; // Thoát hàm sớm vì không có gì để ước lượng
    } // Kết thúc kiểm tra

    // Kiểm tra camera đã được hiệu chuẩn chưa
    if (!isCameraCalibrated()) { // Nếu ma trận camera là ma trận đơn vị (chưa hiệu chuẩn)
        // In cảnh báo ra console
        std::cerr << "[ArucoDetector] CANH BAO: Camera chua duoc hieu chuan! " // Thông báo phần 1
                  << "Ket qua pose se khong chinh xac." << std::endl;           // Thông báo phần 2
        // Không return ở đây - vẫn cố ước lượng với ma trận đơn vị (kết quả gần đúng)
    } // Kết thúc kiểm tra hiệu chuẩn

    // Khai báo vector lưu kết quả ước lượng pose cho tất cả marker
    std::vector<cv::Vec3d> rvecs; // Vector lưu rotation vector (hướng quay 3D) của mỗi marker
    std::vector<cv::Vec3d> tvecs; // Vector lưu translation vector (vị trí 3D) của mỗi marker

    try { // Bắt đầu khối try để xử lý ngoại lệ
        // Gọi estimatePoseSingleMarkers của OpenCV để ước lượng tư thế
        // Hàm này sử dụng thuật toán PnP (Perspective-n-Point) để tính pose
        // Đầu vào: tọa độ góc 2D + kích thước thực marker + thông số camera
        // Đầu ra: rvecs (hướng quay) + tvecs (vị trí) cho mỗi marker
        cv::aruco::estimatePoseSingleMarkers(
            marker_corners_,           // Tọa độ 4 góc 2D của mỗi marker (từ detectMarkers)
            marker_length,             // Kích thước cạnh thực tế của marker (mét), cần chính xác để tính khoảng cách
            camera_matrix_,            // Ma trận nội tại camera (focal length, principal point)
            dist_coeffs_,              // Hệ số méo camera (hiệu chỉnh biến dạng ống kính)
            rvecs,                     // [OUTPUT] Vector chứa rotation vector cho mỗi marker
            tvecs                      // [OUTPUT] Vector chứa translation vector cho mỗi marker
        ); // Kết thúc gọi estimatePoseSingleMarkers

        // Cập nhật thông tin pose vào MarkerInfo tương ứng
        for (size_t i = 0; i < detected_markers_.size(); ++i) { // Lặp qua tất cả marker đã phát hiện
            detected_markers_[i].rvec = rvecs[i];       // Gán rotation vector cho marker thứ i
            detected_markers_[i].tvec = tvecs[i];       // Gán translation vector cho marker thứ i
            detected_markers_[i].pose_valid = true;     // Đánh dấu pose đã được ước lượng thành công

            // Tính khoảng cách Euclidean từ camera đến marker
            // distance = ||tvec|| = sqrt(tx^2 + ty^2 + tz^2) - norm L2 của translation vector
            detected_markers_[i].distance = cv::norm(tvecs[i]); // Tính norm (độ dài) của vector tịnh tiến
        } // Kết thúc vòng lặp cập nhật pose

        pose_estimated_ = true; // Đặt cờ xác nhận pose đã được ước lượng thành công

        // In thông báo thành công
        std::cout << "[ArucoDetector] Da uoc luong pose cho " << detected_markers_.size() // In số marker
                  << " marker(s)" << std::endl;                                             // Kết thúc thông báo
    }
    catch (const cv::Exception& e) { // Bắt ngoại lệ nếu estimatePoseSingleMarkers gặp lỗi
        // In chi tiết lỗi ra stderr
        std::cerr << "[ArucoDetector] LOI khi uoc luong pose: " << e.what() << std::endl; // Thông báo lỗi
    } // Kết thúc catch
} // Kết thúc estimatePose

// ============================================================================
// PHƯƠNG THỨC TÍNH KHOẢNG CÁCH - CALCULATE DISTANCE
// ============================================================================

void ArucoDetector::calculateDistance(float marker_length) { // Tính khoảng cách từ camera đến marker
    // Kiểm tra pose đã được ước lượng chưa
    if (!pose_estimated_) { // Nếu cờ pose_estimated_ = false (chưa gọi estimatePose)
        // Tự động gọi estimatePose trước khi tính khoảng cách
        // Đây là thiết kế "lazy evaluation" - chỉ tính khi cần
        estimatePose(marker_length); // Gọi estimatePose để tính rvec, tvec trước
    } // Kết thúc kiểm tra

    // Duyệt qua tất cả marker đã phát hiện để tính khoảng cách
    for (auto& marker : detected_markers_) { // Sử dụng auto& để tham chiếu trực tiếp (tránh sao chép)
        if (marker.pose_valid) { // Chỉ tính khoảng cách nếu pose hợp lệ (đã ước lượng thành công)
            // Tính khoảng cách Euclidean = norm L2 của translation vector
            // tvec = (tx, ty, tz) là vị trí marker trong hệ tọa độ camera
            // distance = sqrt(tx^2 + ty^2 + tz^2) - khoảng cách thẳng từ camera đến marker
            marker.distance = cv::norm(marker.tvec); // cv::norm tính norm L2 mặc định

            // In khoảng cách ra console để debug
            std::cout << "[ArucoDetector] Marker ID=" << marker.id           // In ID marker
                      << " - Khoang cach: " << marker.distance               // In khoảng cách
                      << " met" << std::endl;                                 // Đơn vị là mét
        } // Kết thúc kiểm tra pose_valid
    } // Kết thúc vòng lặp duyệt marker
} // Kết thúc calculateDistance

// ============================================================================
// PHƯƠNG THỨC VẼ KẾT QUẢ - DRAW DETECTIONS
// ============================================================================

void ArucoDetector::drawDetections(cv::Mat& frame) { // Vẽ kết quả phát hiện lên khung hình
    // Kiểm tra khung hình có hợp lệ không
    if (frame.empty()) { // Nếu ảnh rỗng
        std::cerr << "[ArucoDetector] CANH BAO: Khung hinh rong, khong the ve!" << std::endl; // Cảnh báo
        return; // Thoát hàm sớm
    } // Kết thúc kiểm tra

    // Kiểm tra có marker nào để vẽ không
    if (marker_ids_.empty()) { // Nếu không có marker nào
        return; // Thoát hàm sớm - không cần vẽ gì
    } // Kết thúc kiểm tra

    // ========================================================================
    // BƯỚC 1: VẼ VIỀN MARKER VÀ ID SỬ DỤNG HÀM TÍCH HỢP CỦA OPENCV
    // ========================================================================

    // Sử dụng drawDetectedMarkers của OpenCV để vẽ viền xanh lá quanh mỗi marker và hiển thị ID
    // Hàm này vẽ đường nối 4 góc marker và ghi ID ở góc trên-trái
    cv::aruco::drawDetectedMarkers(frame, marker_corners_, marker_ids_); // Vẽ viền + ID lên frame

    // ========================================================================
    // BƯỚC 2: VẼ TRỤC TỌA ĐỘ 3D VÀ THÔNG TIN BỔ SUNG
    // ========================================================================

    // Duyệt qua từng marker để vẽ thêm thông tin chi tiết
    for (size_t i = 0; i < detected_markers_.size(); ++i) { // Lặp qua tất cả marker
        const auto& marker = detected_markers_[i]; // Tham chiếu const đến marker thứ i để đọc thông tin

        // Nếu pose đã được ước lượng, vẽ trục tọa độ 3D (XYZ axes)
        if (marker.pose_valid) { // Kiểm tra pose có hợp lệ không
            // Vẽ trục tọa độ 3D tại vị trí marker
            // Trục X = đỏ, Trục Y = xanh lá, Trục Z = xanh dương
            // Tham số cuối (0.03) là chiều dài trục hiển thị (3cm)
            cv::drawFrameAxes(
                frame,                 // Ảnh đầu ra để vẽ trục lên
                camera_matrix_,        // Ma trận camera cần để chiếu 3D -> 2D
                dist_coeffs_,          // Hệ số méo camera
                marker.rvec,           // Rotation vector của marker (hướng quay)
                marker.tvec,           // Translation vector của marker (vị trí)
                0.03f                  // Chiều dài trục hiển thị = 3cm (đơn vị mét)
            ); // Kết thúc drawFrameAxes
        } // Kết thúc kiểm tra pose_valid

        // ====================================================================
        // VẼ KHOẢNG CÁCH VÀ THÔNG TIN TEXT
        // ====================================================================

        // Kiểm tra marker có ít nhất 1 góc để xác định vị trí vẽ text
        if (!marker.corners.empty()) { // Nếu có tọa độ góc
            // Tính tâm marker bằng trung bình 4 góc
            cv::Point2f center(0, 0); // Khởi tạo tọa độ tâm tại gốc (0,0)

            // Cộng tọa độ tất cả 4 góc
            for (const auto& corner : marker.corners) { // Duyệt qua 4 góc
                center.x += corner.x; // Cộng tọa độ x của góc
                center.y += corner.y; // Cộng tọa độ y của góc
            } // Kết thúc vòng lặp cộng góc

            // Chia cho 4 để lấy trung bình (tâm marker)
            center.x /= static_cast<float>(marker.corners.size()); // Trung bình x = tổng x / số góc
            center.y /= static_cast<float>(marker.corners.size()); // Trung bình y = tổng y / số góc

            // Nếu có khoảng cách, hiển thị lên ảnh
            if (marker.distance > 0.0) { // Chỉ vẽ nếu khoảng cách đã được tính (> 0)
                // Xây dựng chuỗi hiển thị khoảng cách
                std::ostringstream oss;                             // Tạo string stream để format chuỗi
                oss.precision(2);                                    // Đặt độ chính xác 2 chữ số thập phân
                oss << std::fixed << marker.distance << "m";        // Format: "1.23m" (fixed-point, 2 decimal)
                std::string dist_text = oss.str();                   // Chuyển stream thành string

                // Tính vị trí vẽ text (phía dưới tâm marker, lệch xuống 30 pixel)
                cv::Point text_pos(                                  // Tạo điểm vị trí text
                    static_cast<int>(center.x) - 30,                 // X: lệch trái 30 pixel so với tâm
                    static_cast<int>(center.y) + 30                  // Y: lệch xuống 30 pixel so với tâm
                ); // Kết thúc tạo text_pos

                // Vẽ nền đen (shadow) cho text để dễ đọc trên mọi nền
                cv::putText(
                    frame,                         // Ảnh để vẽ text lên
                    dist_text,                     // Chuỗi cần vẽ (khoảng cách)
                    text_pos + cv::Point(1, 1),    // Vị trí lệch 1 pixel (tạo hiệu ứng bóng)
                    cv::FONT_HERSHEY_SIMPLEX,      // Font chữ đơn giản
                    0.6,                           // Cỡ chữ (scale factor)
                    cv::Scalar(0, 0, 0),           // Màu đen (BGR: 0,0,0) cho bóng
                    2                              // Độ dày nét chữ = 2 pixel
                ); // Kết thúc putText bóng

                // Vẽ text chính màu vàng (dễ nhìn trên nền tối hoặc sáng)
                cv::putText(
                    frame,                         // Ảnh để vẽ text lên
                    dist_text,                     // Chuỗi cần vẽ (khoảng cách)
                    text_pos,                      // Vị trí vẽ text (tại tâm marker lệch xuống)
                    cv::FONT_HERSHEY_SIMPLEX,      // Font chữ đơn giản
                    0.6,                           // Cỡ chữ (scale factor)
                    cv::Scalar(0, 255, 255),       // Màu vàng (BGR: 0,255,255 = Yellow)
                    2                              // Độ dày nét chữ = 2 pixel
                ); // Kết thúc putText chính
            } // Kết thúc kiểm tra khoảng cách

            // ================================================================
            // ĐÁNH DẤU ĐẶC BIỆT CHO MARKER BÃI ĐÁP
            // ================================================================

            // Kiểm tra marker hiện tại có phải là bãi đáp không
            if (marker.id == landing_pad_id_) { // So sánh ID marker với ID bãi đáp
                // Vẽ text "LANDING PAD" phía trên tâm marker
                cv::Point landing_text_pos(                          // Tạo vị trí text bãi đáp
                    static_cast<int>(center.x) - 60,                 // X: lệch trái 60 pixel
                    static_cast<int>(center.y) - 40                  // Y: lệch lên 40 pixel
                ); // Kết thúc tạo vị trí

                // Vẽ nền đen cho text bãi đáp
                cv::putText(
                    frame,                         // Ảnh đầu ra
                    "LANDING PAD",                 // Nhãn bãi đáp
                    landing_text_pos + cv::Point(1, 1), // Vị trí bóng
                    cv::FONT_HERSHEY_SIMPLEX,      // Font chữ
                    0.7,                           // Cỡ chữ lớn hơn text khoảng cách
                    cv::Scalar(0, 0, 0),           // Màu đen cho bóng
                    2                              // Độ dày nét
                ); // Kết thúc vẽ bóng

                // Vẽ text chính "LANDING PAD" màu xanh lá nổi bật
                cv::putText(
                    frame,                         // Ảnh đầu ra
                    "LANDING PAD",                 // Nhãn bãi đáp
                    landing_text_pos,              // Vị trí text chính
                    cv::FONT_HERSHEY_SIMPLEX,      // Font chữ
                    0.7,                           // Cỡ chữ
                    cv::Scalar(0, 255, 0),         // Màu xanh lá (BGR: 0,255,0 = Green)
                    2                              // Độ dày nét
                ); // Kết thúc vẽ text bãi đáp

                // Vẽ hình tròn tại tâm marker bãi đáp để đánh dấu điểm hạ cánh
                cv::circle(
                    frame,                                         // Ảnh đầu ra
                    cv::Point(static_cast<int>(center.x),          // Tâm hình tròn = tâm marker (x)
                              static_cast<int>(center.y)),         // Tâm hình tròn = tâm marker (y)
                    15,                                            // Bán kính 15 pixel
                    cv::Scalar(0, 255, 0),                         // Màu xanh lá
                    3                                              // Độ dày đường viền = 3 pixel
                ); // Kết thúc vẽ hình tròn bãi đáp
            } // Kết thúc kiểm tra bãi đáp
        } // Kết thúc kiểm tra corners không rỗng
    } // Kết thúc vòng lặp duyệt marker

    // Vẽ thống kê tổng số marker ở góc trên-trái ảnh
    std::string count_text = "Markers: " + std::to_string(detected_markers_.size()); // Tạo chuỗi đếm marker
    cv::putText(
        frame,                             // Ảnh đầu ra
        count_text,                        // Chuỗi "Markers: N"
        cv::Point(10, 30),                 // Vị trí góc trên-trái (10, 30)
        cv::FONT_HERSHEY_SIMPLEX,          // Font chữ đơn giản
        0.8,                               // Cỡ chữ
        cv::Scalar(255, 255, 0),           // Màu cyan (BGR: 255,255,0)
        2                                  // Độ dày nét
    ); // Kết thúc vẽ text thống kê
} // Kết thúc drawDetections

// ============================================================================
// PHƯƠNG THỨC TẠO MARKER - GENERATE MARKER
// ============================================================================

cv::Mat ArucoDetector::generateMarker(
    int id,            // ID của marker cần tạo (phải nằm trong phạm vi dictionary)
    int size,          // Kích thước ảnh đầu ra (pixel, ảnh vuông size x size)
    int dict_type      // Loại dictionary (-1 = dùng dictionary hiện tại)
) { // Phương thức tạo hình ảnh ArUco marker
    cv::Mat marker_image; // Khai báo biến Mat để chứa ảnh marker đầu ra

    try { // Bắt đầu khối try để xử lý lỗi
        // Xác định dictionary sử dụng
        cv::Ptr<cv::aruco::Dictionary> dict_to_use; // Con trỏ đến dictionary sẽ dùng

        if (dict_type >= 0) { // Nếu dict_type được chỉ định (>= 0)
            // Lấy dictionary theo loại chỉ định
            dict_to_use = cv::aruco::getPredefinedDictionary(dict_type); // Lấy dictionary mới
            std::cout << "[ArucoDetector] Tao marker voi dictionary loai: " << dict_type << std::endl; // Thông báo
        }
        else { // Nếu dict_type = -1 (mặc định)
            // Sử dụng dictionary hiện tại của detector
            dict_to_use = dictionary_; // Dùng dictionary đã thiết lập trước đó
            std::cout << "[ArucoDetector] Tao marker voi dictionary hien tai" << std::endl; // Thông báo
        } // Kết thúc kiểm tra dict_type

        // Tạo ảnh marker sử dụng OpenCV 4.x API
        // generateImageMarker tạo ảnh nhị phân (đen trắng) của marker với ID cho trước
        // Hàm này thay thế drawMarker đã deprecated trong OpenCV mới hơn
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7 // Kiểm tra version OpenCV >= 4.7
        // OpenCV 4.7+ sử dụng generateImageMarker (API mới)
        cv::aruco::generateImageMarker(
            dict_to_use,       // Dictionary chứa marker
            id,                // ID của marker cần tạo
            size,              // Kích thước ảnh (pixel)
            marker_image,      // [OUTPUT] Ảnh marker đầu ra
            1                  // Border bits = 1 (số bit viền trắng quanh marker)
        ); // Kết thúc generateImageMarker
#else // Nếu OpenCV < 4.7
        // OpenCV cũ hơn sử dụng drawMarker (API cũ, vẫn hoạt động)
        cv::aruco::drawMarker(
            dict_to_use,       // Dictionary chứa marker
            id,                // ID của marker cần tạo
            size,              // Kích thước ảnh (pixel)
            marker_image,      // [OUTPUT] Ảnh marker đầu ra
            1                  // Border bits = 1
        ); // Kết thúc drawMarker
#endif // Kết thúc kiểm tra version

        // In thông báo tạo marker thành công
        std::cout << "[ArucoDetector] Da tao marker ID=" << id     // In ID marker
                  << " kich thuoc " << size << "x" << size         // In kích thước
                  << " pixel" << std::endl;                         // Kết thúc thông báo
    }
    catch (const cv::Exception& e) { // Bắt ngoại lệ nếu tạo marker thất bại
        // In chi tiết lỗi
        std::cerr << "[ArucoDetector] LOI khi tao marker ID=" << id // Thông báo lỗi
                  << ": " << e.what() << std::endl;                  // Chi tiết lỗi
        return cv::Mat(); // Trả về ảnh rỗng nếu thất bại
    } // Kết thúc catch

    return marker_image; // Trả về ảnh marker đã tạo thành công
} // Kết thúc generateMarker

// ============================================================================
// PHƯƠNG THỨC KIỂM TRA BÃI ĐÁP - IS LANDING PAD DETECTED
// ============================================================================

bool ArucoDetector::isLandingPadDetected() const { // Kiểm tra bãi đáp có trong khung hình không
    // Sử dụng std::any_of để tìm kiếm hiệu quả trong danh sách marker
    // Lambda function kiểm tra mỗi marker có ID trùng với landing_pad_id_ không
    return std::any_of(
        detected_markers_.begin(),     // Iterator đầu danh sách marker
        detected_markers_.end(),       // Iterator cuối danh sách marker
        [this](const MarkerInfo& m) {  // Lambda function: capture 'this' để truy cập landing_pad_id_
            return m.id == landing_pad_id_; // So sánh ID marker với ID bãi đáp
        } // Kết thúc lambda
    ); // Kết thúc std::any_of - trả về true nếu tìm thấy ít nhất 1 marker bãi đáp
} // Kết thúc isLandingPadDetected

// ============================================================================
// PHƯƠNG THỨC LẤY POSE BÃI ĐÁP - GET LANDING PAD POSE
// ============================================================================

MarkerInfo ArucoDetector::getLandingPadPose() const { // Lấy thông tin tư thế marker bãi đáp
    // Sử dụng std::find_if để tìm marker bãi đáp trong danh sách
    auto it = std::find_if(
        detected_markers_.begin(),     // Iterator đầu danh sách
        detected_markers_.end(),       // Iterator cuối danh sách
        [this](const MarkerInfo& m) {  // Lambda: tìm marker có ID = landing_pad_id_
            return m.id == landing_pad_id_; // Điều kiện: ID trùng với ID bãi đáp
        } // Kết thúc lambda
    ); // Kết thúc find_if

    // Kiểm tra có tìm thấy marker bãi đáp không
    if (it != detected_markers_.end()) { // Nếu iterator không phải end() => đã tìm thấy
        // In thông tin pose bãi đáp
        std::cout << "[ArucoDetector] Landing Pad pose - Distance: " << it->distance // In khoảng cách
                  << "m" << std::endl;                                                // Đơn vị mét
        return *it; // Trả về MarkerInfo của marker bãi đáp (giải tham chiếu iterator)
    } // Kết thúc kiểm tra tìm thấy

    // Nếu không tìm thấy, in cảnh báo và trả về MarkerInfo mặc định
    std::cerr << "[ArucoDetector] CANH BAO: Khong tim thay Landing Pad!" << std::endl; // Cảnh báo
    return MarkerInfo(); // Trả về MarkerInfo mặc định (id=-1, pose_valid=false)
} // Kết thúc getLandingPadPose

// ============================================================================
// PHƯƠNG THỨC TRUY VẤN - GETTERS
// ============================================================================

const std::vector<MarkerInfo>& ArucoDetector::getDetectedMarkers() const { // Lấy danh sách marker đã phát hiện
    return detected_markers_; // Trả về tham chiếu const đến vector marker (không sao chép, hiệu quả)
} // Kết thúc getDetectedMarkers

size_t ArucoDetector::getMarkerCount() const { // Lấy số lượng marker phát hiện được
    return detected_markers_.size(); // Trả về kích thước vector (số marker)
} // Kết thúc getMarkerCount

// ============================================================================
// PHƯƠNG THỨC PRIVATE - KIỂM TRA HIỆU CHUẨN CAMERA
// ============================================================================

bool ArucoDetector::isCameraCalibrated() const { // Kiểm tra camera đã hiệu chuẩn chưa
    // So sánh ma trận camera với ma trận đơn vị
    // Nếu camera chưa hiệu chuẩn, ma trận camera vẫn là ma trận đơn vị (giá trị mặc định)
    cv::Mat identity = cv::Mat::eye(3, 3, CV_64F); // Tạo ma trận đơn vị 3x3 để so sánh

    // Tính norm (khoảng cách) giữa ma trận camera và ma trận đơn vị
    // Nếu norm > epsilon nhỏ => ma trận camera đã được thay đổi (đã hiệu chuẩn)
    double diff = cv::norm(camera_matrix_, identity, cv::NORM_L2); // Tính norm L2 của hiệu hai ma trận

    // Trả về true nếu sự khác biệt > ngưỡng nhỏ (1e-6), tức là đã hiệu chuẩn
    return diff > 1e-6; // So sánh với epsilon rất nhỏ để tránh sai số số học floating-point
} // Kết thúc isCameraCalibrated

} // namespace drone_vision - Kết thúc namespace drone_vision
