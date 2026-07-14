/**
 * @file aruco_detector.hpp
 * @brief Module phát hiện và xử lý ArUco marker cho hệ thống DroneVisionPro
 * @details File header định nghĩa lớp ArucoDetector với đầy đủ chức năng:
 *          - Phát hiện ArUco marker trong khung hình camera
 *          - Ước lượng tư thế 3D (pose estimation) của marker
 *          - Tính khoảng cách từ camera đến marker
 *          - Vẽ kết quả phát hiện lên khung hình
 *          - Tạo hình ảnh ArUco marker
 *          - Phát hiện bãi đáp (landing pad) cho drone
 * @author Tran Ngoc Bao - 24021238
 * @date 2026-07-13
 */

#pragma once // Đảm bảo file header chỉ được include một lần duy nhất, tránh lỗi định nghĩa trùng lặp

// ============================================================================
// PHẦN INCLUDE CÁC THƯ VIỆN CẦN THIẾT
// ============================================================================

#include <opencv2/opencv.hpp>          // Thư viện OpenCV chính, chứa các hàm xử lý ảnh cơ bản (Mat, imread, imshow, ...)
#include <opencv2/aruco.hpp>           // Module ArUco của OpenCV, cung cấp API phát hiện và xử lý ArUco marker
#include <opencv2/calib3d.hpp>         // Module hiệu chuẩn camera và tái tạo 3D (solvePnP, drawFrameAxes, ...)

#include <vector>                      // Container vector của STL, dùng để lưu danh sách marker, corners, ...
#include <string>                      // Kiểu chuỗi string của STL, dùng cho tên, thông báo, ...
#include <map>                         // Container map của STL, dùng cho ánh xạ key-value
#include <memory>                      // Smart pointer (shared_ptr, unique_ptr) để quản lý bộ nhớ an toàn
#include <cmath>                       // Các hàm toán học (sqrt, pow, ...) dùng tính khoảng cách

// ============================================================================
// ĐỊNH NGHĨA NAMESPACE VÀ CẤU TRÚC DỮ LIỆU
// ============================================================================

namespace drone_vision { // Namespace chính của dự án DroneVisionPro, nhóm tất cả các lớp liên quan

/**
 * @struct MarkerInfo
 * @brief Cấu trúc lưu trữ thông tin đầy đủ của một ArUco marker đã phát hiện
 * @details Mỗi marker phát hiện được sẽ có một MarkerInfo tương ứng chứa:
 *          - ID định danh duy nhất của marker
 *          - Tọa độ 4 góc trên ảnh 2D
 *          - Vector quay và tịnh tiến biểu diễn tư thế 3D
 *          - Khoảng cách ước tính từ camera đến marker
 */
struct MarkerInfo {
    int id;                            // ID định danh duy nhất của marker (0, 1, 2, ..., tùy theo dictionary)
    std::vector<cv::Point2f> corners;  // Vector chứa tọa độ 4 góc của marker trên ảnh 2D (theo chiều kim đồng hồ từ góc trên-trái)
    cv::Vec3d rvec;                    // Rotation vector (vector quay Rodrigues) - biểu diễn hướng quay 3D của marker so với camera
    cv::Vec3d tvec;                    // Translation vector (vector tịnh tiến) - biểu diễn vị trí 3D của marker trong hệ tọa độ camera
    double distance;                   // Khoảng cách ước tính từ camera đến tâm marker (đơn vị mét), tính bằng norm của tvec
    bool pose_valid;                   // Cờ cho biết tư thế 3D đã được ước lượng thành công hay chưa (true = hợp lệ)

    /**
     * @brief Constructor mặc định khởi tạo MarkerInfo với giá trị ban đầu
     * @details Khởi tạo tất cả các trường về giá trị mặc định an toàn:
     *          id = -1 (chưa xác định), distance = 0, pose_valid = false
     */
    MarkerInfo()                       // Constructor mặc định không tham số
        : id(-1)                       // Gán ID = -1 để đánh dấu marker chưa được xác định
        , rvec(0, 0, 0)               // Khởi tạo rotation vector về gốc (không quay)
        , tvec(0, 0, 0)               // Khởi tạo translation vector về gốc tọa độ
        , distance(0.0)               // Khoảng cách ban đầu = 0 (chưa tính)
        , pose_valid(false)           // Tư thế chưa được ước lượng nên đánh dấu không hợp lệ
    {} // Kết thúc constructor mặc định
};

// ============================================================================
// ĐỊNH NGHĨA LỚP CHÍNH ARUCODETECTOR
// ============================================================================

/**
 * @class ArucoDetector
 * @brief Lớp chính phát hiện và xử lý ArUco marker cho ứng dụng drone
 * @details Lớp này cung cấp toàn bộ pipeline xử lý ArUco marker:
 *          1. Phát hiện marker trong khung hình (detect)
 *          2. Ước lượng tư thế 3D (estimatePose)
 *          3. Tính khoảng cách (calculateDistance)
 *          4. Vẽ kết quả trực quan (drawDetections)
 *          5. Kiểm tra bãi đáp (isLandingPadDetected)
 *          6. Tạo marker mới (generateMarker)
 */
class ArucoDetector {
public: // Phần public - các phương thức và thuộc tính có thể truy cập từ bên ngoài lớp

    // ========================================================================
    // CONSTRUCTOR VÀ DESTRUCTOR
    // ========================================================================

    /**
     * @brief Constructor chính của ArucoDetector
     * @param camera_matrix Ma trận nội tại camera 3x3 (chứa focal length và principal point)
     *                      Mặc định là ma trận đơn vị 3x3 nếu chưa hiệu chuẩn camera
     * @param dist_coeffs Vector hệ số méo (distortion coefficients) của camera
     *                    Mặc định là vector rỗng nếu chưa hiệu chuẩn camera
     * @details Constructor khởi tạo detector với thông số camera. Nếu không truyền
     *          tham số, sử dụng giá trị mặc định (ma trận đơn vị) cho camera chưa hiệu chuẩn.
     *          Dictionary mặc định là DICT_4X4_50 (dictionary 4x4 bit với 50 marker).
     */
    explicit ArucoDetector(                                      // Từ khóa explicit ngăn chuyển đổi ngầm định
        const cv::Mat& camera_matrix = cv::Mat::eye(3, 3, CV_64F), // Ma trận camera mặc định là ma trận đơn vị 3x3 kiểu double
        const cv::Mat& dist_coeffs = cv::Mat::zeros(5, 1, CV_64F) // Hệ số méo mặc định là vector 0 (5 phần tử, không méo)
    ); // Kết thúc khai báo constructor

    /**
     * @brief Destructor mặc định
     * @details Sử dụng destructor mặc định của compiler vì không có tài nguyên
     *          cần giải phóng thủ công (smart pointer tự quản lý bộ nhớ)
     */
    ~ArucoDetector() = default; // Destructor mặc định, compiler tự tạo code giải phóng bộ nhớ

    // ========================================================================
    // CÁC PHƯƠNG THỨC CẤU HÌNH (CONFIGURATION METHODS)
    // ========================================================================

    /**
     * @brief Thiết lập loại dictionary ArUco sử dụng
     * @param dict_type Kiểu dictionary (ví dụ: cv::aruco::DICT_4X4_50, cv::aruco::DICT_5X5_100, ...)
     * @details Dictionary xác định bộ marker nào sẽ được sử dụng để phát hiện.
     *          Các loại phổ biến:
     *          - DICT_4X4_50:   50 marker, mỗi marker 4x4 bit (nhỏ, nhanh)
     *          - DICT_5X5_100: 100 marker, mỗi marker 5x5 bit (cân bằng)
     *          - DICT_6X6_250: 250 marker, mỗi marker 6x6 bit (nhiều marker)
     *          - DICT_7X7_1000: 1000 marker, mỗi marker 7x7 bit (rất nhiều marker)
     */
    void setDictionary(int dict_type); // Phương thức thiết lập dictionary, nhận tham số kiểu int tương ứng với enum PredefinedDictionaryType

    /**
     * @brief Thiết lập ID của marker dùng làm bãi đáp (landing pad)
     * @param id ID của marker bãi đáp (mặc định là 0)
     * @details Marker bãi đáp là marker đặc biệt đánh dấu vị trí hạ cánh an toàn cho drone.
     *          Khi phát hiện marker này, drone có thể tự động hạ cánh chính xác.
     */
    void setLandingPadId(int id); // Phương thức đặt ID bãi đáp, dùng cho chức năng tự động hạ cánh

    /**
     * @brief Cập nhật ma trận camera và hệ số méo
     * @param camera_matrix Ma trận nội tại camera 3x3 mới
     * @param dist_coeffs Vector hệ số méo mới
     * @details Gọi phương thức này sau khi hiệu chuẩn camera để cập nhật thông số mới.
     *          Ma trận camera chính xác giúp ước lượng tư thế 3D chính xác hơn.
     */
    void setCameraParameters(                                  // Phương thức cập nhật thông số camera
        const cv::Mat& camera_matrix,                          // Ma trận nội tại camera mới (3x3, CV_64F)
        const cv::Mat& dist_coeffs                             // Vector hệ số méo mới (5x1 hoặc 4x1, CV_64F)
    ); // Kết thúc khai báo setCameraParameters

    // ========================================================================
    // CÁC PHƯƠNG THỨC PHÁT HIỆN VÀ XỬ LÝ (DETECTION & PROCESSING METHODS)
    // ========================================================================

    /**
     * @brief Phát hiện tất cả ArUco marker trong khung hình
     * @param frame Khung hình đầu vào từ camera (cv::Mat, BGR hoặc grayscale)
     * @return Vector chứa thông tin MarkerInfo của tất cả marker phát hiện được
     * @details Phương thức chính để phát hiện marker. Quy trình:
     *          1. Chuyển ảnh sang grayscale (nếu cần)
     *          2. Gọi cv::aruco::detectMarkers để tìm marker
     *          3. Tạo MarkerInfo cho mỗi marker tìm được
     *          4. Lưu kết quả vào biến thành viên detected_markers_
     */
    std::vector<MarkerInfo> detect(const cv::Mat& frame); // Phát hiện marker, trả về vector MarkerInfo

    /**
     * @brief Ước lượng tư thế 3D (pose) của tất cả marker đã phát hiện
     * @param marker_length Kích thước cạnh thực tế của marker (đơn vị: mét)
     * @details Phương thức này sử dụng cv::aruco::estimatePoseSingleMarkers để tính
     *          rotation vector (rvec) và translation vector (tvec) cho mỗi marker.
     *          Yêu cầu: phải gọi detect() trước, và camera phải được hiệu chuẩn.
     *          Kết quả pose được lưu vào trường rvec, tvec của mỗi MarkerInfo.
     */
    void estimatePose(float marker_length); // Ước lượng tư thế 3D, cần biết kích thước thực của marker

    /**
     * @brief Tính khoảng cách từ camera đến mỗi marker đã phát hiện
     * @param marker_length Kích thước cạnh thực tế của marker (đơn vị: mét)
     * @details Khoảng cách được tính bằng norm (độ dài) của translation vector:
     *          distance = sqrt(tvec[0]^2 + tvec[1]^2 + tvec[2]^2)
     *          Nếu pose chưa được ước lượng, tự động gọi estimatePose() trước.
     */
    void calculateDistance(float marker_length); // Tính khoảng cách, tự động gọi estimatePose nếu cần

    /**
     * @brief Vẽ kết quả phát hiện lên khung hình
     * @param frame Khung hình đầu ra (sẽ được vẽ trực tiếp lên, pass by reference)
     * @details Vẽ các thông tin trực quan lên ảnh:
     *          - Viền marker với màu sắc
     *          - ID của marker
     *          - Trục tọa độ 3D (nếu có pose)
     *          - Khoảng cách (nếu đã tính)
     *          - Đánh dấu đặc biệt cho marker bãi đáp
     */
    void drawDetections(cv::Mat& frame); // Vẽ kết quả lên ảnh, thay đổi trực tiếp frame

    /**
     * @brief Tạo hình ảnh ArUco marker với ID và kích thước cho trước
     * @param id ID của marker cần tạo (phải nằm trong phạm vi của dictionary)
     * @param size Kích thước ảnh marker đầu ra (pixel, ảnh vuông size x size)
     * @param dict_type Loại dictionary sử dụng (mặc định -1 = dùng dictionary hiện tại)
     * @return Ảnh marker dạng cv::Mat (grayscale, đen trắng)
     * @details Hữu ích để in marker ra giấy hoặc hiển thị trên màn hình.
     *          Sử dụng cv::aruco::generateImageMarker (OpenCV 4.x) hoặc
     *          cv::aruco::drawMarker (OpenCV cũ hơn) để tạo marker.
     */
    cv::Mat generateMarker(                                    // Tạo ảnh marker, trả về cv::Mat chứa ảnh
        int id,                                                // ID của marker cần tạo
        int size = 200,                                        // Kích thước ảnh đầu ra (mặc định 200x200 pixel)
        int dict_type = -1                                     // Loại dictionary (-1 = dùng dictionary hiện tại)
    ); // Kết thúc khai báo generateMarker

    // ========================================================================
    // CÁC PHƯƠNG THỨC KIỂM TRA BÃI ĐÁP (LANDING PAD METHODS)
    // ========================================================================

    /**
     * @brief Kiểm tra xem marker bãi đáp có được phát hiện không
     * @return true nếu marker có ID = landing_pad_id_ được phát hiện, false nếu không
     * @details Duyệt qua danh sách marker đã phát hiện, tìm marker có ID trùng
     *          với landing_pad_id_ (mặc định là 0). Dùng cho chức năng tự động hạ cánh.
     */
    bool isLandingPadDetected() const; // Phương thức const vì không thay đổi trạng thái đối tượng

    /**
     * @brief Lấy thông tin tư thế của marker bãi đáp
     * @return MarkerInfo của marker bãi đáp (nếu phát hiện được)
     * @details Trả về MarkerInfo đầy đủ của marker bãi đáp bao gồm:
     *          vị trí, hướng, khoảng cách. Nếu không tìm thấy, trả về MarkerInfo mặc định.
     */
    MarkerInfo getLandingPadPose() const; // Lấy pose bãi đáp, const vì chỉ đọc dữ liệu

    // ========================================================================
    // CÁC PHƯƠNG THỨC TRUY VẤN (GETTER METHODS)
    // ========================================================================

    /**
     * @brief Lấy danh sách tất cả marker đã phát hiện
     * @return Tham chiếu const đến vector MarkerInfo
     * @details Trả về tham chiếu để tránh sao chép dữ liệu không cần thiết.
     *          Vector này được cập nhật mỗi khi gọi detect().
     */
    const std::vector<MarkerInfo>& getDetectedMarkers() const; // Trả về const reference để hiệu quả và an toàn

    /**
     * @brief Lấy số lượng marker đã phát hiện
     * @return Số lượng marker trong lần detect() gần nhất
     */
    size_t getMarkerCount() const; // Trả về size_t vì số lượng luôn >= 0

private: // Phần private - các thuộc tính và phương thức chỉ truy cập được từ bên trong lớp

    // ========================================================================
    // BIẾN THÀNH VIÊN PRIVATE
    // ========================================================================

    cv::Mat camera_matrix_;            // Ma trận nội tại camera 3x3 (fx, fy, cx, cy) - dùng cho ước lượng pose
    cv::Mat dist_coeffs_;              // Vector hệ số méo camera (k1, k2, p1, p2, k3) - hiệu chỉnh biến dạng ống kính

    cv::Ptr<cv::aruco::Dictionary> dictionary_;       // Con trỏ thông minh đến dictionary ArUco đang sử dụng
    cv::Ptr<cv::aruco::DetectorParameters> parameters_; // Con trỏ thông minh đến tham số cấu hình detector

    std::vector<MarkerInfo> detected_markers_;         // Vector lưu trữ thông tin marker đã phát hiện trong lần detect() gần nhất

    std::vector<std::vector<cv::Point2f>> marker_corners_; // Vector 2D lưu tọa độ 4 góc của mỗi marker (dùng nội bộ cho OpenCV API)
    std::vector<int> marker_ids_;                          // Vector lưu ID của mỗi marker phát hiện được (dùng nội bộ)
    std::vector<std::vector<cv::Point2f>> rejected_candidates_; // Vector lưu các vùng bị loại bỏ (không phải marker hợp lệ)

    int landing_pad_id_;               // ID của marker bãi đáp (mặc định = 0), dùng cho chức năng tự động hạ cánh
    bool pose_estimated_;              // Cờ đánh dấu tư thế 3D đã được ước lượng hay chưa trong chu kỳ hiện tại

    // ========================================================================
    // PHƯƠNG THỨC PRIVATE HỖ TRỢ
    // ========================================================================

    /**
     * @brief Kiểm tra ma trận camera có hợp lệ không
     * @return true nếu camera đã được hiệu chuẩn (ma trận camera khác ma trận đơn vị)
     * @details Camera chưa hiệu chuẩn sẽ có ma trận đơn vị, không thể ước lượng pose chính xác.
     */
    bool isCameraCalibrated() const; // Kiểm tra hiệu chuẩn camera, const vì chỉ đọc

    /**
     * @brief Vẽ hình khối lập phương ảo 3D đè lên mặt phẳng của marker
     * @param frame Khung hình đầu ra
     * @param rvec Rotation vector
     * @param tvec Translation vector
     * @param marker_length Kích thước thật của marker (mét)
     */
    void drawCube(cv::Mat& frame, const cv::Vec3d& rvec, const cv::Vec3d& tvec, float marker_length) const;
};

} // namespace drone_vision - Kết thúc namespace drone_vision
