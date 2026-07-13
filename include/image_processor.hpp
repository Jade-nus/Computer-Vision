/**
 * @file image_processor.hpp
 * @brief Lớp xử lý ảnh chuyên dụng cho drone với các thuật toán tối ưu cho ảnh chụp từ trên cao
 * @details Module này cung cấp các chức năng xử lý ảnh thiết yếu cho drone:
 *          - Cải thiện chất lượng ảnh (CLAHE, khử sương mù, làm nét, khử nhiễu)
 *          - Phát hiện cạnh (Canny, Sobel, Laplacian)
 *          - Lọc màu sắc trong không gian HSV
 *          - Phép toán hình thái học (erosion, dilation, opening, closing)
 *          - Ổn định khung hình bằng optical flow
 *          - Zoom kỹ thuật số với tâm tùy chỉnh
 * @author Tran Ngoc Bao - 24021238
 * @date 2026
 */

#pragma once // Đảm bảo file header chỉ được biên dịch một lần, tránh lỗi multiple definition

// === CÁC THƯ VIỆN OPENCV CẦN THIẾT ===
#include <opencv2/opencv.hpp>          // Thư viện OpenCV cốt lõi: Mat, Point, Scalar, Rect,...
#include <opencv2/imgproc.hpp>         // Module xử lý ảnh: filter, morphology, color conversion,...
#include <opencv2/photo.hpp>           // Module xử lý ảnh nâng cao: denoising, HDR, inpainting,...
#include <opencv2/video/tracking.hpp>  // Module tracking: optical flow, Kalman filter,...

// === CÁC THƯ VIỆN CHUẨN C++ ===
#include <vector>    // Container động std::vector cho danh sách điểm đặc trưng
#include <algorithm> // Thuật toán chuẩn: min, max, sort, transform,...
#include <cmath>     // Hàm toán học: sqrt, pow, abs, atan2,...

/**
 * @namespace drone_vision
 * @brief Không gian tên chứa tất cả module thị giác máy tính cho drone
 */
namespace drone_vision {

    /**
     * @enum EdgeMethod
     * @brief Liệt kê các phương pháp phát hiện cạnh được hỗ trợ
     * @details Mỗi phương pháp có ưu nhược điểm riêng:
     *          - CANNY: Phổ biến nhất, cho cạnh mỏng và rõ ràng, sử dụng 2 ngưỡng
     *          - SOBEL: Tính đạo hàm bậc 1, cho gradient theo hướng x hoặc y
     *          - LAPLACIAN: Tính đạo hàm bậc 2, nhạy với nhiễu nhưng phát hiện tốt
     */
    enum EdgeMethod {
        EDGE_CANNY     = 0,   // Phương pháp Canny - phát hiện cạnh đa ngưỡng, phổ biến nhất
        EDGE_SOBEL     = 1,   // Phương pháp Sobel - tính gradient bậc 1 theo x và y
        EDGE_LAPLACIAN = 2    // Phương pháp Laplacian - tính đạo hàm bậc 2 (Laplacian)
    };

    /**
     * @enum MorphOperation
     * @brief Liệt kê các phép toán hình thái học được hỗ trợ
     * @details Phép toán hình thái học dùng để xử lý hình dạng trong ảnh nhị phân:
     *          - ERODE: Co nhỏ vùng trắng, loại bỏ nhiễu nhỏ
     *          - DILATE: Mở rộng vùng trắng, lấp đầy lỗ nhỏ
     *          - OPEN: Erode rồi Dilate, loại nhiễu giữ hình dạng
     *          - CLOSE: Dilate rồi Erode, lấp lỗ giữ hình dạng
     *          - GRADIENT: Dilate - Erode, tìm viền đối tượng
     *          - TOPHAT: Src - Opening, tìm chi tiết sáng nhỏ
     *          - BLACKHAT: Closing - Src, tìm chi tiết tối nhỏ
     */
    enum MorphOperation {
        MORPH_ERODE_OP    = 0,   // Phép co (erosion) - thu nhỏ vùng trắng trong ảnh
        MORPH_DILATE_OP   = 1,   // Phép giãn (dilation) - mở rộng vùng trắng trong ảnh
        MORPH_OPEN_OP     = 2,   // Phép mở (opening) = erode + dilate, loại nhiễu nhỏ
        MORPH_CLOSE_OP    = 3,   // Phép đóng (closing) = dilate + erode, lấp lỗ nhỏ
        MORPH_GRADIENT_OP = 4,   // Gradient hình thái = dilate - erode, tìm đường viền
        MORPH_TOPHAT_OP   = 5,   // Top-hat = src - opening, tìm chi tiết sáng trên nền tối
        MORPH_BLACKHAT_OP = 6    // Black-hat = closing - src, tìm chi tiết tối trên nền sáng
    };

    /**
     * @class ImageProcessor
     * @brief Lớp xử lý ảnh chuyên dụng cho ứng dụng drone
     * @details Cung cấp bộ công cụ xử lý ảnh đầy đủ được tối ưu cho ảnh chụp từ drone:
     *          - Ảnh drone thường bị ảnh hưởng bởi sương mù, rung lắc, độ cao
     *          - Các thuật toán được điều chỉnh để phù hợp với đặc thù ảnh aerial
     *          - Hỗ trợ cả ảnh màu (BGR) và ảnh xám (grayscale)
     * 
     * Ví dụ sử dụng:
     * @code
     *   drone_vision::ImageProcessor processor;
     *   cv::Mat enhanced = processor.applyCLAHE(frame);
     *   cv::Mat dehazed = processor.dehaze(frame, 0.7f);
     *   cv::Mat edges = processor.detectEdges(frame, drone_vision::EDGE_CANNY);
     * @endcode
     */
    class ImageProcessor {

    public:
        /**
         * @brief Constructor - khởi tạo bộ xử lý ảnh
         * @details Thiết lập các tham số mặc định cho xử lý ảnh drone
         */
        ImageProcessor(); // Khởi tạo ImageProcessor với cấu hình mặc định

        /**
         * @brief Destructor - giải phóng tài nguyên
         */
        ~ImageProcessor(); // Giải phóng bộ nhớ và tài nguyên

        // ========================================================================
        // CẢI THIỆN CHẤT LƯỢNG ẢNH (IMAGE ENHANCEMENT)
        // ========================================================================

        /**
         * @brief Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
         * @param src Ảnh đầu vào (BGR hoặc grayscale)
         * @param clip_limit Giới hạn tương phản (2.0 là mặc định, tăng = tương phản mạnh hơn)
         * @param tile_size Kích thước ô lưới để tính histogram cục bộ (8x8 là mặc định)
         * @return Ảnh đã được cân bằng histogram thích ứng, cùng kích thước và kiểu với ảnh gốc
         * @details CLAHE tốt hơn histogram equalization thông thường vì:
         *          - Tính histogram cục bộ trên từng ô nhỏ, không bị ảnh hưởng bởi vùng sáng/tối lớn
         *          - Clip limit giới hạn khuếch đại nhiễu ở vùng đồng nhất
         *          - Đặc biệt hiệu quả cho ảnh drone chụp từ trên cao với ánh sáng không đều
         */
        cv::Mat applyCLAHE(const cv::Mat& src,                             // Ảnh đầu vào cần cân bằng histogram
                           double clip_limit = 2.0,                        // Giới hạn khuếch đại tương phản cục bộ
                           cv::Size tile_size = cv::Size(8, 8));           // Kích thước ô lưới tính histogram

        /**
         * @brief Khử sương mù cho ảnh chụp từ drone (dehazing)
         * @param src Ảnh đầu vào bị sương mù (BGR, 8-bit)
         * @param strength Cường độ khử sương (0.0 = không khử, 1.0 = khử tối đa), mặc định 0.7
         * @return Ảnh đã được khử sương mù, màu sắc tự nhiên hơn
         * @details Sử dụng phương pháp Dark Channel Prior đơn giản hóa:
         *          1. Tính dark channel (giá trị min trên 3 kênh màu trong vùng lân cận)
         *          2. Ước lượng ánh sáng khí quyển (atmospheric light) từ vùng sáng nhất
         *          3. Ước lượng bản đồ truyền dẫn (transmission map)
         *          4. Khôi phục ảnh gốc từ mô hình sương mù
         *          Đặc biệt quan trọng cho drone bay ở độ cao lớn hoặc điều kiện thời tiết xấu
         */
        cv::Mat dehaze(const cv::Mat& src,         // Ảnh bị sương mù cần xử lý
                       float strength = 0.7f);     // Cường độ khử sương từ 0.0 đến 1.0

        /**
         * @brief Điều chỉnh độ sáng và tương phản của ảnh
         * @param src Ảnh đầu vào (BGR hoặc grayscale)
         * @param alpha Hệ số tương phản (1.0 = giữ nguyên, >1.0 = tăng, <1.0 = giảm)
         * @param beta Giá trị cộng thêm cho độ sáng (-255 đến 255, 0 = giữ nguyên)
         * @return Ảnh đã điều chỉnh: output(x,y) = alpha * src(x,y) + beta
         * @details Công thức: g(x,y) = α * f(x,y) + β
         *          α (alpha) kiểm soát tương phản, β (beta) kiểm soát độ sáng
         */
        cv::Mat adjustBrightnessContrast(const cv::Mat& src,     // Ảnh gốc cần điều chỉnh
                                         double alpha = 1.0,     // Hệ số nhân tương phản (gain)
                                         int beta = 0);          // Giá trị cộng độ sáng (bias)

        /**
         * @brief Làm nét ảnh bằng phương pháp Unsharp Masking
         * @param src Ảnh đầu vào cần làm nét (BGR)
         * @return Ảnh đã được làm nét, chi tiết rõ ràng hơn
         * @details Unsharp masking hoạt động theo nguyên lý:
         *          1. Tạo ảnh mờ (Gaussian blur) từ ảnh gốc
         *          2. Trừ ảnh mờ từ ảnh gốc để lấy chi tiết (unsharp mask)
         *          3. Cộng chi tiết ngược lại vào ảnh gốc với hệ số khuếch đại
         *          Kết quả: sharpened = original + amount * (original - blurred)
         */
        cv::Mat sharpen(const cv::Mat& src); // Làm nét ảnh bằng unsharp masking

        /**
         * @brief Khử nhiễu ảnh chụp từ drone
         * @param src Ảnh đầu vào bị nhiễu (BGR, 8-bit)
         * @return Ảnh đã khử nhiễu, giữ được chi tiết cạnh
         * @details Sử dụng cv::fastNlMeansDenoisingColored cho ảnh màu
         *          Thuật toán Non-Local Means tìm và trung bình hóa các pixel tương tự
         *          trong vùng lân cận rộng, hiệu quả hơn bộ lọc Gaussian đơn giản
         *          Quan trọng cho ảnh drone do rung lắc và ISO cao ở điều kiện thiếu sáng
         */
        cv::Mat denoise(const cv::Mat& src); // Khử nhiễu bằng Non-Local Means

        // ========================================================================
        // PHÁT HIỆN CẠNH (EDGE DETECTION)
        // ========================================================================

        /**
         * @brief Phát hiện cạnh trong ảnh bằng nhiều phương pháp khác nhau
         * @param src Ảnh đầu vào (BGR hoặc grayscale)
         * @param method Phương pháp phát hiện cạnh: 0=Canny, 1=Sobel, 2=Laplacian
         * @return Ảnh cạnh (grayscale, 8-bit)
         * @details Mỗi phương pháp có đặc điểm:
         *          - Canny: Cạnh mỏng, rõ ràng, dùng 2 ngưỡng (hysteresis), tốt cho đa số trường hợp
         *          - Sobel: Gradient hướng, cho biết cường độ và hướng cạnh, tốt cho phân tích hướng
         *          - Laplacian: Đạo hàm bậc 2, phát hiện cạnh mạnh nhưng nhạy nhiễu
         */
        cv::Mat detectEdges(const cv::Mat& src,     // Ảnh đầu vào cần phát hiện cạnh
                            int method = 0);        // Phương pháp: 0=Canny, 1=Sobel, 2=Laplacian

        // ========================================================================
        // LỌC MÀU SẮC (COLOR FILTERING)
        // ========================================================================

        /**
         * @brief Lọc đối tượng theo màu sắc trong không gian màu HSV
         * @param src Ảnh đầu vào (BGR, 8-bit)
         * @param lower_hsv Giá trị HSV thấp nhất của dải màu cần lọc (H: 0-179, S: 0-255, V: 0-255)
         * @param upper_hsv Giá trị HSV cao nhất của dải màu cần lọc
         * @return Ảnh BGR chỉ chứa các pixel có màu nằm trong dải cho trước, phần còn lại = đen
         * @details Không gian HSV (Hue-Saturation-Value) phù hợp lọc màu hơn BGR vì:
         *          - H (Hue): Sắc độ màu, ít bị ảnh hưởng bởi ánh sáng
         *          - S (Saturation): Độ bão hòa, phân biệt màu sắc và xám
         *          - V (Value): Độ sáng, tách biệt với sắc độ
         *          Ví dụ lọc màu đỏ: lower=(0,100,100), upper=(10,255,255)
         */
        cv::Mat filterByColor(const cv::Mat& src,             // Ảnh BGR đầu vào
                              cv::Scalar lower_hsv,           // Giới hạn dưới HSV (Hue, Saturation, Value)
                              cv::Scalar upper_hsv);          // Giới hạn trên HSV (Hue, Saturation, Value)

        // ========================================================================
        // PHÉP TOÁN HÌNH THÁI HỌC (MORPHOLOGICAL OPERATIONS)
        // ========================================================================

        /**
         * @brief Áp dụng phép toán hình thái học lên ảnh
         * @param src Ảnh đầu vào (grayscale hoặc binary)
         * @param operation Loại phép toán (xem enum MorphOperation)
         * @param kernel_size Kích thước phần tử cấu trúc (structuring element), mặc định 5x5
         * @return Ảnh sau khi áp dụng phép toán hình thái học
         * @details Phần tử cấu trúc hình ellipse được sử dụng mặc định vì:
         *          - Cho kết quả mượt mà hơn hình chữ nhật
         *          - Phù hợp với hầu hết đối tượng trong ảnh drone
         */
        cv::Mat applyMorphology(const cv::Mat& src,        // Ảnh đầu vào (nên là binary hoặc grayscale)
                                int operation,             // Loại phép toán hình thái (enum MorphOperation)
                                int kernel_size = 5);      // Kích thước kernel (số lẻ: 3, 5, 7, 9,...)

        // ========================================================================
        // ỔN ĐỊNH KHUNG HÌNH (FRAME STABILIZATION)
        // ========================================================================

        /**
         * @brief Ổn định khung hình giữa 2 frame liên tiếp bằng optical flow
         * @param prev Khung hình trước đó (BGR)
         * @param curr Khung hình hiện tại cần ổn định (BGR)
         * @return Khung hình hiện tại đã được ổn định (bù chuyển động camera/drone)
         * @details Quy trình ổn định:
         *          1. Tìm đặc trưng tốt (good features) trên frame trước bằng Shi-Tomasi
         *          2. Theo dõi đặc trưng sang frame hiện tại bằng Lucas-Kanade optical flow
         *          3. Ước lượng phép biến đổi affine giữa 2 tập điểm
         *          4. Áp dụng biến đổi ngược lên frame hiện tại để bù chuyển động
         *          Đặc biệt quan trọng cho drone vì rung lắc do gió và động cơ
         */
        cv::Mat stabilize(const cv::Mat& prev,    // Khung hình trước (tham chiếu)
                          const cv::Mat& curr);   // Khung hình hiện tại cần ổn định

        // ========================================================================
        // ZOOM KỸ THUẬT SỐ (DIGITAL ZOOM)
        // ========================================================================

        /**
         * @brief Zoom kỹ thuật số vào vùng quan tâm của ảnh
         * @param src Ảnh đầu vào (BGR)
         * @param zoom_factor Hệ số zoom (1.0 = không zoom, 2.0 = phóng to 2x, tối đa 10.0)
         * @param center Tâm zoom (mặc định Point2f(-1,-1) = tâm ảnh)
         * @return Ảnh đã zoom, cùng kích thước với ảnh gốc
         * @details Cắt vùng ảnh quanh tâm zoom với kích thước = original_size / zoom_factor
         *          rồi phóng to về kích thước gốc bằng nội suy INTER_LINEAR
         *          Lưu ý: zoom kỹ thuật số không tăng độ phân giải thực tế
         */
        cv::Mat digitalZoom(const cv::Mat& src,                                // Ảnh gốc cần zoom
                            float zoom_factor,                                  // Hệ số phóng đại (>1.0)
                            cv::Point2f center = cv::Point2f(-1.0f, -1.0f));   // Tâm zoom, (-1,-1)=giữa ảnh

    private:
        // === THÀNH VIÊN PRIVATE ===

        /**
         * @brief Tính Dark Channel của ảnh cho thuật toán khử sương mù
         * @param src Ảnh đầu vào BGR (3 kênh)
         * @param patch_size Kích thước vùng lân cận để tính min (mặc định 15)
         * @return Ảnh dark channel (grayscale, 1 kênh)
         * @details Dark channel = min trên 3 kênh BGR, rồi min filter trên vùng lân cận
         *          Dựa trên quan sát: trong ảnh không có sương, hầu hết vùng ảnh
         *          đều có ít nhất 1 kênh màu có giá trị rất thấp (gần 0)
         */
        cv::Mat computeDarkChannel(const cv::Mat& src,     // Ảnh BGR đầu vào
                                   int patch_size = 15);   // Kích thước vùng lân cận

        /**
         * @brief Ước lượng ánh sáng khí quyển từ ảnh bị sương mù
         * @param src Ảnh BGR đầu vào bị sương mù
         * @param dark_channel Dark channel đã tính trước
         * @return Giá trị ánh sáng khí quyển dạng Scalar (B, G, R)
         * @details Chọn top 0.1% pixel sáng nhất trong dark channel
         *          rồi lấy giá trị trung bình tương ứng trên ảnh gốc
         *          Ánh sáng khí quyển đại diện cho màu sương mù
         */
        cv::Scalar estimateAtmosphericLight(const cv::Mat& src,              // Ảnh gốc BGR
                                            const cv::Mat& dark_channel);    // Dark channel đã tính
    };

} // namespace drone_vision - Kết thúc không gian tên drone_vision
