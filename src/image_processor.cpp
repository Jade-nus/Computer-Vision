/**
 * @file image_processor.cpp
 * @brief Triển khai đầy đủ lớp ImageProcessor với các thuật toán xử lý ảnh cho drone
 * @details File này chứa toàn bộ implementation của class ImageProcessor:
 *          - Cải thiện chất lượng ảnh: CLAHE, dehaze, sharpen, denoise
 *          - Phát hiện cạnh: Canny, Sobel, Laplacian
 *          - Lọc màu sắc HSV, phép toán hình thái học
 *          - Ổn định khung hình bằng optical flow
 *          - Zoom kỹ thuật số với tâm tùy chỉnh
 * @author Tran Ngoc Bao - 24021238
 * @date 2026
 */

#include "image_processor.hpp" // Include header chứa khai báo lớp ImageProcessor, enums, và phương thức

// Include thêm thư viện chuẩn cho implementation
#include <iostream>   // Xuất/nhập console cho thông báo lỗi và debug
#include <numeric>    // Hàm tính toán số học: iota, accumulate,...

/**
 * @namespace drone_vision
 * @brief Không gian tên chứa toàn bộ module thị giác máy tính cho drone
 */
namespace drone_vision {

    // ========================================================================
    // CONSTRUCTOR VÀ DESTRUCTOR
    // ========================================================================

    /**
     * @brief Constructor - khởi tạo bộ xử lý ảnh với cấu hình mặc định
     */
    ImageProcessor::ImageProcessor()
    {
        // Constructor không cần khởi tạo gì đặc biệt
        // Tất cả các method đều là stateless (không lưu trạng thái giữa các lần gọi)
        // trừ stabilize() cần 2 frame liên tiếp (được truyền qua tham số)
        std::cout << "[ImageProcessor] Da khoi tao bo xu ly anh cho drone." << std::endl;
    }

    /**
     * @brief Destructor - giải phóng tài nguyên
     */
    ImageProcessor::~ImageProcessor()
    {
        // Không có tài nguyên động cần giải phóng thủ công
        // Tất cả cv::Mat tự động giải phóng nhờ reference counting nội bộ
    }

    // ========================================================================
    // CẢI THIỆN CHẤT LƯỢNG ẢNH (IMAGE ENHANCEMENT)
    // ========================================================================

    /**
     * @brief Áp dụng CLAHE - Cân bằng histogram thích ứng có giới hạn tương phản
     */
    cv::Mat ImageProcessor::applyCLAHE(const cv::Mat& src,           // Ảnh đầu vào (BGR hoặc grayscale)
                                        double clip_limit,           // Giới hạn khuếch đại tương phản
                                        cv::Size tile_size)          // Kích thước ô lưới tính histogram
    {
        // Kiểm tra ảnh đầu vào có hợp lệ không
        if (src.empty()) { // Ảnh rỗng = không có dữ liệu pixel nào
            std::cerr << "[ImageProcessor] Loi: Anh dau vao rong cho CLAHE." << std::endl;
            return src; // Trả về ảnh rỗng gốc
        }

        cv::Mat result; // Biến lưu kết quả CLAHE

        // Xử lý khác nhau tùy theo ảnh là grayscale hay màu
        if (src.channels() == 1) { // Ảnh grayscale (1 kênh)
            // Tạo đối tượng CLAHE với clip limit và tile size cho trước
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, tile_size);
            // cv::Ptr là smart pointer của OpenCV, tự quản lý bộ nhớ
            // clip_limit giới hạn khuếch đại histogram cục bộ để tránh amplify noise
            // tile_size chia ảnh thành lưới ô nhỏ, mỗi ô tính histogram riêng

            // Áp dụng CLAHE lên ảnh grayscale
            clahe->apply(src, result); // src → CLAHE → result (cải thiện tương phản cục bộ)

        } else if (src.channels() == 3) { // Ảnh màu BGR (3 kênh)
            // Chuyển BGR sang LAB color space
            // LAB tách riêng kênh sáng (L) và kênh màu (A, B)
            // Chỉ áp dụng CLAHE lên kênh L để cải thiện tương phản mà không thay đổi màu sắc
            cv::Mat lab_image; // Biến tạm chứa ảnh trong không gian LAB
            cv::cvtColor(src, lab_image, cv::COLOR_BGR2Lab); // Chuyển BGR → LAB

            // Tách ảnh LAB thành 3 kênh riêng biệt
            std::vector<cv::Mat> lab_channels; // Vector chứa 3 kênh L, A, B
            cv::split(lab_image, lab_channels); // Tách: lab_channels[0]=L, [1]=A, [2]=B

            // Tạo đối tượng CLAHE
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, tile_size);
            // Tạo CLAHE processor với tham số người dùng chỉ định

            // Áp dụng CLAHE CHỈ lên kênh L (Lightness - độ sáng)
            // Giữ nguyên kênh A (green-red) và B (blue-yellow) để bảo toàn màu sắc
            clahe->apply(lab_channels[0], lab_channels[0]); // CLAHE trên kênh L tại chỗ

            // Gộp 3 kênh L, A, B lại thành ảnh LAB hoàn chỉnh
            cv::merge(lab_channels, lab_image); // Gộp: L + A + B → LAB

            // Chuyển ngược từ LAB về BGR để hiển thị bình thường
            cv::cvtColor(lab_image, result, cv::COLOR_Lab2BGR); // LAB → BGR

        } else {
            // Trường hợp ảnh có số kênh khác (ví dụ: BGRA 4 kênh)
            std::cerr << "[ImageProcessor] Canh bao: So kenh anh khong ho tro cho CLAHE: "
                      << src.channels() << std::endl;
            return src.clone(); // Trả về bản sao ảnh gốc không xử lý
        }

        return result; // Trả về ảnh đã cải thiện tương phản
    }

    /**
     * @brief Khử sương mù cho ảnh drone bằng Dark Channel Prior đơn giản hóa
     */
    cv::Mat ImageProcessor::dehaze(const cv::Mat& src,     // Ảnh bị sương mù (BGR, 8-bit)
                                    float strength)        // Cường độ khử sương (0.0 - 1.0)
    {
        // Kiểm tra ảnh đầu vào
        if (src.empty() || src.channels() != 3) { // Ảnh rỗng hoặc không phải 3 kênh
            std::cerr << "[ImageProcessor] Loi: Anh dau vao khong hop le cho dehaze." << std::endl;
            return src; // Trả về ảnh gốc nếu không hợp lệ
        }

        // Giới hạn strength trong khoảng [0.0, 1.0]
        strength = std::max(0.0f, std::min(1.0f, strength)); // Clamp giá trị vào phạm vi hợp lệ

        // Bước 1: Chuyển ảnh sang kiểu float [0, 1] để tính toán chính xác
        cv::Mat src_float; // Biến tạm chứa ảnh dạng float
        src.convertTo(src_float, CV_64FC3, 1.0 / 255.0); // Chia 255 để normalize về [0, 1]

        // Bước 2: Tính Dark Channel của ảnh
        // Dark channel dựa trên quan sát: trong ảnh không sương, ít nhất 1 kênh có giá trị thấp
        cv::Mat dark_channel = computeDarkChannel(src, 15); // Tính dark channel với patch size 15x15

        // Bước 3: Ước lượng ánh sáng khí quyển (atmospheric light)
        // Đây là giá trị BGR đại diện cho màu của sương mù
        cv::Scalar atmospheric_light = estimateAtmosphericLight(src, dark_channel);
        // Lấy giá trị A = (A_b, A_g, A_r) từ vùng sáng nhất trong dark channel

        // Bước 4: Ước lượng bản đồ truyền dẫn (transmission map)
        // Mô hình sương mù: I(x) = J(x)*t(x) + A*(1-t(x))
        // Trong đó: I = ảnh quan sát, J = ảnh gốc, t = transmission, A = atmospheric light
        // transmission t(x) = 1 - strength * dark_channel_normalized(x)

        // Chuẩn bị dark channel dạng float
        cv::Mat dark_float; // Dark channel dạng float [0, 1]
        dark_channel.convertTo(dark_float, CV_64F, 1.0 / 255.0); // Normalize dark channel

        // Tính transmission map: t(x) = 1 - omega * (dark_channel / A)
        // omega = strength, điều khiển mức độ khử sương
        // Chia dark channel cho atmospheric light để normalize
        double a_min = std::min({atmospheric_light[0],      // Giá trị Blue của atmospheric light
                                 atmospheric_light[1],      // Giá trị Green
                                 atmospheric_light[2]});    // Giá trị Red
        a_min = std::max(a_min, 1.0); // Đảm bảo không chia cho 0

        // Tính transmission: t = 1 - strength * (dark / A_min)
        cv::Mat transmission = 1.0 - strength * (dark_float / (a_min / 255.0));
        // transmission gần 1 ở vùng không sương, gần 0 ở vùng sương dày

        // Giới hạn transmission tối thiểu = 0.1 để tránh chia cho 0 và khuếch đại nhiễu
        double t_min = 0.1; // Ngưỡng transmission tối thiểu
        cv::max(transmission, t_min, transmission); // Clamp: t = max(t, 0.1)

        // Bước 5: Khôi phục ảnh gốc (scene radiance recovery)
        // Công thức: J(x) = (I(x) - A) / max(t(x), t0) + A
        cv::Mat result = cv::Mat::zeros(src_float.size(), src_float.type()); // Tạo ảnh kết quả kích thước giống gốc

        // Tách 3 kênh BGR của ảnh float
        std::vector<cv::Mat> channels; // Vector chứa 3 kênh B, G, R
        cv::split(src_float, channels); // Tách: channels[0]=B, [1]=G, [2]=R

        // Khôi phục từng kênh màu riêng biệt
        std::vector<cv::Mat> result_channels(3); // Vector chứa 3 kênh kết quả

        for (int c = 0; c < 3; ++c) { // Duyệt qua 3 kênh B, G, R
            // J(c) = (I(c) - A(c)/255) / t + A(c)/255
            double a_normalized = atmospheric_light[c] / 255.0; // Normalize atmospheric light cho kênh c

            // Áp dụng công thức khôi phục
            result_channels[c] = (channels[c] - a_normalized); // I(c) - A(c): trừ atmospheric light
            cv::divide(result_channels[c], transmission, result_channels[c]); // Chia cho transmission map
            result_channels[c] = result_channels[c] + a_normalized; // Cộng lại atmospheric light
        }

        // Gộp 3 kênh kết quả lại thành ảnh BGR
        cv::merge(result_channels, result); // B + G + R → BGR

        // Chuyển kết quả về kiểu 8-bit [0, 255]
        cv::Mat output; // Ảnh kết quả cuối cùng 8-bit
        result.convertTo(output, CV_8UC3, 255.0); // Nhân 255 và chuyển về uint8

        // Clamp giá trị pixel về [0, 255] để tránh overflow/underflow
        cv::min(output, cv::Scalar(255, 255, 255), output); // Giới hạn max = 255
        cv::max(output, cv::Scalar(0, 0, 0), output);       // Giới hạn min = 0

        return output; // Trả về ảnh đã khử sương mù
    }

    /**
     * @brief Điều chỉnh độ sáng và tương phản
     */
    cv::Mat ImageProcessor::adjustBrightnessContrast(const cv::Mat& src,    // Ảnh gốc
                                                      double alpha,         // Hệ số tương phản (gain)
                                                      int beta)             // Giá trị sáng cộng thêm (bias)
    {
        // Kiểm tra ảnh đầu vào
        if (src.empty()) { // Ảnh rỗng
            std::cerr << "[ImageProcessor] Loi: Anh dau vao rong cho adjustBrightnessContrast." << std::endl;
            return src; // Trả về ảnh rỗng
        }

        cv::Mat result; // Biến lưu kết quả

        // Áp dụng công thức: output(x,y) = saturate(alpha * src(x,y) + beta)
        // convertTo thực hiện phép nhân + cộng + saturate cast trong một bước
        // alpha kiểm soát tương phản: >1 tăng, <1 giảm, =1 giữ nguyên
        // beta kiểm soát độ sáng: >0 sáng hơn, <0 tối hơn, =0 giữ nguyên
        src.convertTo(result,           // Ảnh đích lưu kết quả
                      -1,              // -1 = giữ nguyên kiểu dữ liệu (CV_8U)
                      alpha,           // Hệ số nhân (contrast gain)
                      beta);           // Giá trị cộng thêm (brightness bias)
        // saturate_cast tự động clamp giá trị về [0, 255] cho CV_8U

        return result; // Trả về ảnh đã điều chỉnh
    }

    /**
     * @brief Làm nét ảnh bằng phương pháp Unsharp Masking
     */
    cv::Mat ImageProcessor::sharpen(const cv::Mat& src) // Ảnh cần làm nét (BGR)
    {
        // Kiểm tra ảnh đầu vào
        if (src.empty()) { // Ảnh rỗng
            std::cerr << "[ImageProcessor] Loi: Anh dau vao rong cho sharpen." << std::endl;
            return src; // Trả về ảnh rỗng
        }

        // Bước 1: Tạo ảnh mờ (blurred) bằng Gaussian blur
        cv::Mat blurred; // Biến tạm chứa ảnh mờ
        cv::GaussianBlur(src,                  // Ảnh đầu vào
                         blurred,              // Ảnh đầu ra mờ
                         cv::Size(0, 0),       // Kích thước kernel = 0 → tự tính từ sigma
                         3.0);                 // Sigma = 3.0 (độ mờ Gaussian, sigma lớn = mờ nhiều)
        // GaussianBlur với sigma=3.0 tạo phiên bản mờ mượt của ảnh gốc
        // Kernel size = 0 nghĩa OpenCV tự tính kernel size từ sigma: k = round(6*sigma + 1)|odd

        // Bước 2: Áp dụng Unsharp Masking
        // Công thức: sharpened = src + amount * (src - blurred)
        // Trong đó amount = 1.5 (hệ số khuếch đại chi tiết)
        double amount = 1.5; // Hệ số khuếch đại cho unsharp mask (1.0-3.0 phù hợp)

        cv::Mat sharpened; // Biến lưu kết quả làm nét

        // addWeighted thực hiện: dst = alpha*src1 + beta*src2 + gamma
        // Ở đây: sharpened = (1 + amount) * src + (-amount) * blurred + 0
        //       = src + amount * (src - blurred)
        cv::addWeighted(src,                // Ảnh gốc (src1)
                        1.0 + amount,       // alpha = 1 + amount = 2.5
                        blurred,            // Ảnh mờ (src2)
                        -amount,            // beta = -amount = -1.5
                        0,                  // gamma = 0 (không cộng hằng số)
                        sharpened);         // Ảnh kết quả đã làm nét

        return sharpened; // Trả về ảnh đã được làm nét
    }

    /**
     * @brief Khử nhiễu ảnh bằng Non-Local Means Denoising
     */
    cv::Mat ImageProcessor::denoise(const cv::Mat& src) // Ảnh bị nhiễu (BGR, 8-bit)
    {
        // Kiểm tra ảnh đầu vào
        if (src.empty()) { // Ảnh rỗng
            std::cerr << "[ImageProcessor] Loi: Anh dau vao rong cho denoise." << std::endl;
            return src; // Trả về ảnh rỗng
        }

        cv::Mat denoised; // Biến lưu kết quả khử nhiễu

        // Kiểm tra số kênh để chọn hàm khử nhiễu phù hợp
        if (src.channels() == 3) { // Ảnh màu BGR
            // fastNlMeansDenoisingColored cho ảnh màu
            // Thuật toán Non-Local Means tìm các patch tương tự trong vùng lân cận rộng
            // và tính trung bình có trọng số, hiệu quả hơn Gaussian blur đơn giản
            cv::fastNlMeansDenoisingColored(
                src,         // Ảnh đầu vào bị nhiễu
                denoised,    // Ảnh đầu ra đã khử nhiễu
                10,          // h = 10: cường độ lọc kênh luminance (tăng = mượt hơn, mất chi tiết)
                10,          // hColor = 10: cường độ lọc kênh màu (thường bằng h)
                7,           // templateWindowSize = 7: kích thước patch so sánh (7x7 pixel)
                21           // searchWindowSize = 21: kích thước vùng tìm kiếm patch tương tự (21x21)
            );
            // h và hColor kiểm soát mức độ khử nhiễu
            // templateWindowSize: kích thước mẫu dùng để so sánh (nhỏ hơn = nhanh hơn)
            // searchWindowSize: vùng tìm patch tương tự (lớn hơn = chất lượng tốt hơn nhưng chậm hơn)

        } else if (src.channels() == 1) { // Ảnh grayscale
            // fastNlMeansDenoising cho ảnh 1 kênh
            cv::fastNlMeansDenoising(
                src,         // Ảnh grayscale bị nhiễu
                denoised,    // Ảnh grayscale đã khử nhiễu
                10,          // h = 10: cường độ lọc
                7,           // templateWindowSize = 7
                21           // searchWindowSize = 21
            );

        } else { // Ảnh có số kênh không hỗ trợ (vd: 4 kênh BGRA)
            std::cerr << "[ImageProcessor] Canh bao: So kenh khong ho tro cho denoise." << std::endl;
            return src.clone(); // Trả về bản sao không xử lý
        }

        return denoised; // Trả về ảnh đã khử nhiễu
    }

    // ========================================================================
    // PHÁT HIỆN CẠNH (EDGE DETECTION)
    // ========================================================================

    /**
     * @brief Phát hiện cạnh bằng Canny, Sobel hoặc Laplacian
     */
    cv::Mat ImageProcessor::detectEdges(const cv::Mat& src,   // Ảnh đầu vào (BGR hoặc grayscale)
                                         int method)          // Phương pháp: 0=Canny, 1=Sobel, 2=Laplacian
    {
        // Kiểm tra ảnh đầu vào
        if (src.empty()) { // Ảnh rỗng
            std::cerr << "[ImageProcessor] Loi: Anh dau vao rong cho detectEdges." << std::endl;
            return src; // Trả về ảnh rỗng
        }

        // Bước 1: Chuyển ảnh về grayscale nếu đang là ảnh màu
        cv::Mat gray; // Biến chứa ảnh xám
        if (src.channels() == 3) { // Ảnh BGR 3 kênh
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY); // Chuyển BGR → Grayscale
        } else {
            gray = src.clone(); // Đã là grayscale, tạo bản sao
        }

        // Bước 2: Làm mờ nhẹ để giảm nhiễu trước khi phát hiện cạnh
        // Nhiễu gây ra cạnh giả (false edges), đặc biệt nghiêm trọng cho Laplacian
        cv::Mat blurred; // Biến chứa ảnh đã làm mờ
        cv::GaussianBlur(gray,                 // Ảnh xám đầu vào
                         blurred,              // Ảnh xám đã làm mờ
                         cv::Size(5, 5),       // Kernel 5x5 (kích thước vừa phải)
                         1.4);                 // Sigma = 1.4 (mờ nhẹ đủ giảm nhiễu)
        // Gaussian blur là bước tiền xử lý quan trọng cho edge detection

        cv::Mat edges; // Biến chứa ảnh cạnh kết quả

        // Chọn phương pháp phát hiện cạnh
        switch (method) {
            case EDGE_CANNY: { // Phương pháp Canny - phổ biến nhất

                // Tính ngưỡng tự động dựa trên median của ảnh
                // Phương pháp Otsu-inspired: sử dụng giá trị trung vị
                double median_val = 0.0; // Biến lưu giá trị trung vị

                // Tính histogram để tìm median
                int hist_size = 256; // 256 bins cho ảnh 8-bit (giá trị 0-255)
                float range[] = {0, 256}; // Phạm vi giá trị pixel
                const float* hist_range = {range}; // Con trỏ đến phạm vi

                cv::Mat hist; // Ma trận histogram
                cv::calcHist(&blurred,     // Ảnh đầu vào (con trỏ)
                             1,            // Số ảnh = 1
                             nullptr,      // Kênh 0 (mặc định)
                             cv::Mat(),    // Không dùng mask
                             hist,         // Histogram đầu ra
                             1,            // Số chiều histogram = 1
                             &hist_size,   // Số bins = 256
                             &hist_range); // Phạm vi giá trị

                // Tìm giá trị median từ histogram
                int total_pixels = blurred.rows * blurred.cols; // Tổng số pixel
                int count = 0;     // Bộ đếm tích lũy
                for (int i = 0; i < 256; ++i) { // Duyệt qua từng bin
                    count += static_cast<int>(hist.at<float>(i)); // Cộng dồn số pixel
                    if (count >= total_pixels / 2) { // Đạt 50% tổng pixel = vị trí median
                        median_val = static_cast<double>(i); // Ghi nhận giá trị median
                        break; // Thoát vòng lặp khi tìm thấy
                    }
                }

                // Tính ngưỡng Canny tự động từ median
                // Công thức empirical: lower = 0.67 * median, upper = 1.33 * median
                double lower_thresh = std::max(0.0, 0.67 * median_val);   // Ngưỡng dưới
                double upper_thresh = std::min(255.0, 1.33 * median_val); // Ngưỡng trên
                // Canny sử dụng hysteresis thresholding:
                // - Gradient > upper → chắc chắn là cạnh
                // - lower < gradient < upper → là cạnh nếu kết nối với cạnh chắc chắn
                // - Gradient < lower → không phải cạnh

                // Áp dụng Canny edge detection
                cv::Canny(blurred,          // Ảnh đầu vào đã làm mờ
                          edges,            // Ảnh cạnh đầu ra (binary: 0 hoặc 255)
                          lower_thresh,     // Ngưỡng dưới (hysteresis low)
                          upper_thresh);    // Ngưỡng trên (hysteresis high)
                break; // Thoát switch
            }

            case EDGE_SOBEL: { // Phương pháp Sobel - tính gradient

                cv::Mat grad_x, grad_y; // Gradient theo hướng x và y

                // Tính đạo hàm theo hướng x (gradient ngang - phát hiện cạnh dọc)
                cv::Sobel(blurred,          // Ảnh đầu vào
                          grad_x,           // Gradient theo x
                          CV_16S,           // Kiểu đầu ra 16-bit signed (chứa giá trị âm)
                          1,                // Order x = 1 (đạo hàm bậc 1 theo x)
                          0,                // Order y = 0 (không tính đạo hàm theo y)
                          3);               // Kernel size = 3 (Sobel 3x3)
                // CV_16S cần thiết vì Sobel tạo giá trị âm khi gradient đổi hướng

                // Tính đạo hàm theo hướng y (gradient dọc - phát hiện cạnh ngang)
                cv::Sobel(blurred,          // Ảnh đầu vào
                          grad_y,           // Gradient theo y
                          CV_16S,           // Kiểu 16-bit signed
                          0,                // Order x = 0
                          1,                // Order y = 1 (đạo hàm bậc 1 theo y)
                          3);               // Kernel size = 3

                // Chuyển gradient về giá trị tuyệt đối 8-bit
                cv::Mat abs_grad_x, abs_grad_y; // Giá trị tuyệt đối của gradient

                cv::convertScaleAbs(grad_x, abs_grad_x); // |grad_x| chuyển về CV_8U
                cv::convertScaleAbs(grad_y, abs_grad_y); // |grad_y| chuyển về CV_8U
                // convertScaleAbs: dst = saturate(|src * alpha + beta|)
                // Chuyển 16-bit signed sang 8-bit unsigned với absolute value

                // Kết hợp gradient x và y thành gradient tổng hợp
                // Xấp xỉ: magnitude ≈ 0.5*|grad_x| + 0.5*|grad_y|
                cv::addWeighted(abs_grad_x,     // Gradient x tuyệt đối
                                0.5,            // Trọng số 0.5 cho gradient x
                                abs_grad_y,     // Gradient y tuyệt đối
                                0.5,            // Trọng số 0.5 cho gradient y
                                0,              // Gamma = 0
                                edges);         // Gradient tổng hợp (cường độ cạnh)
                // Đây là xấp xỉ của sqrt(gx^2 + gy^2), nhanh hơn tính sqrt thật
                break; // Thoát switch
            }

            case EDGE_LAPLACIAN: { // Phương pháp Laplacian - đạo hàm bậc 2
                cv::Mat laplacian_result; // Kết quả Laplacian thô

                // Tính Laplacian (đạo hàm bậc 2 = tổng đạo hàm bậc 2 theo x và y)
                cv::Laplacian(blurred,              // Ảnh đầu vào đã làm mờ
                              laplacian_result,      // Kết quả Laplacian (16-bit signed)
                              CV_16S,               // Kiểu đầu ra 16-bit signed
                              3);                   // Kernel size = 3
                // Laplacian = d²f/dx² + d²f/dy² (tổng đạo hàm riêng bậc 2)
                // Phát hiện vùng có sự thay đổi gradient (zero-crossing = cạnh)
                // Nhạy với nhiễu hơn Canny và Sobel nên cần Gaussian blur trước

                // Chuyển về giá trị tuyệt đối 8-bit
                cv::convertScaleAbs(laplacian_result, edges); // |laplacian| → 8-bit
                break; // Thoát switch
            }

            default: { // Phương pháp không hợp lệ
                std::cerr << "[ImageProcessor] Canh bao: Phuong phap phat hien canh khong hop le: "
                          << method << ". Su dung Canny mac dinh." << std::endl;

                // Fallback: sử dụng Canny với ngưỡng mặc định
                cv::Canny(blurred,      // Ảnh đầu vào
                          edges,        // Ảnh cạnh đầu ra
                          50,           // Ngưỡng dưới mặc định = 50
                          150);         // Ngưỡng trên mặc định = 150
                break; // Thoát switch
            }
        }

        return edges; // Trả về ảnh cạnh (grayscale/binary tùy phương pháp)
    }

    // ========================================================================
    // LỌC MÀU SẮC (COLOR FILTERING)
    // ========================================================================

    /**
     * @brief Lọc đối tượng theo dải màu trong không gian HSV
     */
    cv::Mat ImageProcessor::filterByColor(const cv::Mat& src,         // Ảnh BGR đầu vào
                                           cv::Scalar lower_hsv,     // HSV giới hạn dưới
                                           cv::Scalar upper_hsv)     // HSV giới hạn trên
    {
        // Kiểm tra ảnh đầu vào
        if (src.empty() || src.channels() != 3) { // Ảnh rỗng hoặc không phải 3 kênh
            std::cerr << "[ImageProcessor] Loi: Anh dau vao khong hop le cho filterByColor." << std::endl;
            return src; // Trả về ảnh gốc
        }

        // Bước 1: Chuyển ảnh từ BGR sang không gian màu HSV
        // HSV phù hợp lọc màu hơn BGR vì tách riêng sắc độ (H) với độ sáng (V)
        cv::Mat hsv_image; // Biến chứa ảnh HSV
        cv::cvtColor(src, hsv_image, cv::COLOR_BGR2HSV); // BGR → HSV
        // Trong OpenCV: H ∈ [0, 179], S ∈ [0, 255], V ∈ [0, 255]
        // Lưu ý: H trong OpenCV = H_thật / 2 (vì 360° không vừa 8-bit)

        // Bước 2: Tạo mask nhị phân cho các pixel có màu trong dải cho trước
        cv::Mat mask; // Mask nhị phân: 255 = trong dải, 0 = ngoài dải
        cv::inRange(hsv_image,     // Ảnh HSV đầu vào
                    lower_hsv,     // Giới hạn dưới (H_min, S_min, V_min)
                    upper_hsv,     // Giới hạn trên (H_max, S_max, V_max)
                    mask);         // Mask đầu ra (binary)
        // inRange: pixel(x,y) = 255 nếu lower ≤ hsv(x,y) ≤ upper cho MỌI kênh

        // Bước 3: Áp dụng mask lên ảnh gốc BGR bằng phép AND theo bit
        cv::Mat result; // Ảnh kết quả chỉ giữ lại pixel có màu phù hợp
        cv::bitwise_and(src,        // Ảnh BGR gốc
                        src,        // Ảnh BGR gốc (lấy giá trị pixel từ đây)
                        result,     // Ảnh kết quả: giữ pixel có mask=255, đen pixel có mask=0
                        mask);      // Mask quyết định pixel nào được giữ
        // bitwise_and với mask: result(x,y) = src(x,y) nếu mask(x,y)=255, ngược lại = 0

        return result; // Trả về ảnh BGR chỉ chứa vùng có màu trong dải HSV
    }

    // ========================================================================
    // PHÉP TOÁN HÌNH THÁI HỌC (MORPHOLOGICAL OPERATIONS)
    // ========================================================================

    /**
     * @brief Áp dụng phép toán hình thái học
     */
    cv::Mat ImageProcessor::applyMorphology(const cv::Mat& src,     // Ảnh đầu vào
                                             int operation,          // Loại phép toán (enum)
                                             int kernel_size)        // Kích thước kernel
    {
        // Kiểm tra ảnh đầu vào
        if (src.empty()) { // Ảnh rỗng
            std::cerr << "[ImageProcessor] Loi: Anh dau vao rong cho applyMorphology." << std::endl;
            return src; // Trả về ảnh rỗng
        }

        // Đảm bảo kernel_size là số lẻ dương (yêu cầu của OpenCV)
        if (kernel_size % 2 == 0) { // Nếu là số chẵn
            kernel_size += 1; // Tăng thêm 1 để thành số lẻ
        }
        kernel_size = std::max(1, kernel_size); // Đảm bảo ít nhất = 1

        // Tạo phần tử cấu trúc (structuring element) hình ellipse
        // Ellipse cho kết quả mượt hơn hình chữ nhật (MORPH_RECT) hoặc chữ thập (MORPH_CROSS)
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,                                    // Hình dạng: ellipse (hình bầu dục)
            cv::Size(kernel_size, kernel_size)                    // Kích thước kernel (ví dụ: 5x5)
        );
        // Structuring element quyết định hình dạng vùng lân cận xét trong phép morphology

        cv::Mat result; // Biến chứa ảnh kết quả

        // Chọn và áp dụng phép toán hình thái phù hợp
        switch (operation) {
            case MORPH_ERODE_OP: // Phép co (erosion)
                // Erosion: pixel = min trong vùng lân cận → thu nhỏ vùng trắng
                cv::erode(src, result, kernel); // Áp dụng erosion với kernel ellipse
                break;

            case MORPH_DILATE_OP: // Phép giãn (dilation)
                // Dilation: pixel = max trong vùng lân cận → mở rộng vùng trắng
                cv::dilate(src, result, kernel); // Áp dụng dilation với kernel ellipse
                break;

            case MORPH_OPEN_OP: // Phép mở (opening = erosion + dilation)
                // Opening loại bỏ nhiễu nhỏ (điểm trắng nhỏ) mà giữ nguyên hình dạng lớn
                cv::morphologyEx(src, result, cv::MORPH_OPEN, kernel); // Erosion rồi Dilation
                break;

            case MORPH_CLOSE_OP: // Phép đóng (closing = dilation + erosion)
                // Closing lấp đầy lỗ nhỏ (điểm đen nhỏ) trong vùng trắng
                cv::morphologyEx(src, result, cv::MORPH_CLOSE, kernel); // Dilation rồi Erosion
                break;

            case MORPH_GRADIENT_OP: // Gradient hình thái (dilation - erosion)
                // Gradient cho đường viền (biên) của đối tượng
                cv::morphologyEx(src, result, cv::MORPH_GRADIENT, kernel); // Dilate - Erode = viền
                break;

            case MORPH_TOPHAT_OP: // Top-hat (original - opening)
                // Top-hat tách chi tiết sáng nhỏ trên nền tối hơn
                cv::morphologyEx(src, result, cv::MORPH_TOPHAT, kernel); // Src - Opening
                break;

            case MORPH_BLACKHAT_OP: // Black-hat (closing - original)
                // Black-hat tách chi tiết tối nhỏ trên nền sáng hơn
                cv::morphologyEx(src, result, cv::MORPH_BLACKHAT, kernel); // Closing - Src
                break;

            default: // Phép toán không hợp lệ
                std::cerr << "[ImageProcessor] Canh bao: Phep toan hinh thai khong hop le: "
                          << operation << ". Su dung Opening mac dinh." << std::endl;
                cv::morphologyEx(src, result, cv::MORPH_OPEN, kernel); // Fallback: Opening
                break;
        }

        return result; // Trả về ảnh sau phép toán hình thái
    }

    // ========================================================================
    // ỔN ĐỊNH KHUNG HÌNH (FRAME STABILIZATION)
    // ========================================================================

    /**
     * @brief Ổn định khung hình giữa 2 frame liên tiếp bằng optical flow
     */
    cv::Mat ImageProcessor::stabilize(const cv::Mat& prev,   // Khung hình trước (BGR)
                                       const cv::Mat& curr)  // Khung hình hiện tại (BGR)
    {
        // Kiểm tra cả 2 ảnh đầu vào
        if (prev.empty() || curr.empty()) { // Một trong hai ảnh rỗng
            std::cerr << "[ImageProcessor] Loi: Anh dau vao rong cho stabilize." << std::endl;
            return curr; // Trả về frame hiện tại không xử lý
        }

        // Kiểm tra kích thước 2 frame phải bằng nhau
        if (prev.size() != curr.size()) { // Kích thước khác nhau
            std::cerr << "[ImageProcessor] Loi: Kich thuoc 2 frame khong khop cho stabilize." << std::endl;
            return curr; // Trả về frame hiện tại không xử lý
        }

        // Bước 1: Chuyển cả 2 frame về grayscale
        // Optical flow chỉ làm việc trên ảnh grayscale
        cv::Mat prev_gray, curr_gray; // Biến chứa ảnh xám

        if (prev.channels() == 3) { // Frame trước là ảnh màu
            cv::cvtColor(prev, prev_gray, cv::COLOR_BGR2GRAY); // BGR → Grayscale
        } else {
            prev_gray = prev.clone(); // Đã là grayscale, clone
        }

        if (curr.channels() == 3) { // Frame hiện tại là ảnh màu
            cv::cvtColor(curr, curr_gray, cv::COLOR_BGR2GRAY); // BGR → Grayscale
        } else {
            curr_gray = curr.clone(); // Đã là grayscale, clone
        }

        // Bước 2: Tìm các đặc trưng tốt (good features) trên frame trước
        // Sử dụng Shi-Tomasi corner detector (goodFeaturesToTrack)
        std::vector<cv::Point2f> prev_points; // Vector chứa tọa độ đặc trưng trên frame trước

        cv::goodFeaturesToTrack(
            prev_gray,          // Ảnh xám của frame trước
            prev_points,        // Danh sách đặc trưng tìm được (output)
            200,                // Số đặc trưng tối đa muốn tìm (200 là đủ cho ổn định)
            0.01,               // Quality level: chỉ lấy corner có quality ≥ 1% so với corner tốt nhất
            30                  // Min distance: khoảng cách tối thiểu giữa 2 corner (30 pixel)
        );
        // goodFeaturesToTrack tìm các góc (corner) mạnh trong ảnh
        // Các góc này dễ theo dõi (track) qua nhiều frame

        // Kiểm tra có tìm đủ đặc trưng không
        if (prev_points.size() < 10) { // Ít hơn 10 đặc trưng = không đủ tin cậy
            std::cerr << "[ImageProcessor] Canh bao: Khong du dac trung de on dinh ("
                      << prev_points.size() << " diem)." << std::endl;
            return curr.clone(); // Trả về bản sao frame hiện tại không ổn định
        }

        // Bước 3: Theo dõi đặc trưng từ frame trước sang frame hiện tại bằng Lucas-Kanade Optical Flow
        std::vector<cv::Point2f> curr_points;  // Vị trí đặc trưng trên frame hiện tại (output)
        std::vector<uchar> status;             // Trạng thái theo dõi: 1=thành công, 0=mất dấu
        std::vector<float> err;                // Sai số theo dõi cho mỗi điểm

        cv::calcOpticalFlowPyrLK(
            prev_gray,          // Ảnh xám frame trước
            curr_gray,          // Ảnh xám frame hiện tại
            prev_points,        // Vị trí đặc trưng trên frame trước (input)
            curr_points,        // Vị trí đặc trưng trên frame hiện tại (output)
            status,             // Trạng thái tracking cho mỗi điểm (output)
            err                 // Sai số tracking cho mỗi điểm (output)
        );
        // calcOpticalFlowPyrLK sử dụng phương pháp Lucas-Kanade với image pyramid
        // Pyramid cho phép theo dõi chuyển động lớn bằng cách xử lý ở nhiều scale

        // Bước 4: Lọc chỉ giữ các điểm đã theo dõi thành công
        std::vector<cv::Point2f> good_prev, good_curr; // Danh sách điểm tốt (tracking thành công)

        for (size_t i = 0; i < status.size(); ++i) { // Duyệt qua tất cả các điểm
            if (status[i]) { // status = 1 nghĩa là tracking thành công cho điểm này
                good_prev.push_back(prev_points[i]); // Thêm vị trí trên frame trước
                good_curr.push_back(curr_points[i]); // Thêm vị trí tương ứng trên frame hiện tại
            }
        }

        // Kiểm tra có đủ điểm tốt để ước lượng biến đổi không
        if (good_prev.size() < 4) { // Cần ít nhất 4 điểm cho affine transform
            std::cerr << "[ImageProcessor] Canh bao: Khong du diem tot de uoc luong bien doi ("
                      << good_prev.size() << " diem)." << std::endl;
            return curr.clone(); // Trả về frame hiện tại không ổn định
        }

        // Bước 5: Ước lượng phép biến đổi affine giữa 2 tập điểm
        // Affine transform mô tả: translation + rotation + scaling + shear
        cv::Mat transform_matrix = cv::estimateAffinePartial2D(
            good_prev,     // Điểm trên frame trước (source)
            good_curr      // Điểm tương ứng trên frame hiện tại (destination)
        );
        // estimateAffinePartial2D ước lượng ma trận 2x3 affine với RANSAC
        // Partial2D = chỉ translation + rotation + uniform scale (4 DOF thay vì 6 DOF)
        // RANSAC loại bỏ outlier (điểm sai do vật thể di chuyển)

        // Kiểm tra ma trận biến đổi có hợp lệ không
        if (transform_matrix.empty()) { // Không ước lượng được (quá nhiều outlier)
            std::cerr << "[ImageProcessor] Canh bao: Khong uoc luong duoc phep bien doi." << std::endl;
            return curr.clone(); // Trả về frame hiện tại không ổn định
        }

        // Bước 6: Áp dụng biến đổi ngược (inverse warp) lên frame hiện tại
        // Biến đổi ngược bù lại chuyển động camera để ổn định ảnh
        cv::Mat stabilized; // Khung hình đã ổn định

        // Tính ma trận nghịch đảo để bù chuyển động
        // Thay vì đảo ngược ma trận, ta sử dụng WARP_INVERSE_MAP flag
        cv::warpAffine(curr,                   // Ảnh nguồn (frame hiện tại)
                       stabilized,             // Ảnh đích (frame đã ổn định)
                       transform_matrix,       // Ma trận biến đổi affine 2x3
                       curr.size(),            // Kích thước đầu ra = kích thước gốc
                       cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, // Nội suy tuyến tính + ánh xạ ngược
                       cv::BORDER_REPLICATE);  // Sao chép pixel biên cho vùng trống
        // WARP_INVERSE_MAP: dùng ma trận nghịch đảo (bù chuyển động thay vì áp dụng chuyển động)
        // BORDER_REPLICATE: lặp lại pixel biên thay vì để đen vùng trống ở rìa

        return stabilized; // Trả về khung hình đã ổn định (bù chuyển động camera/drone)
    }

    // ========================================================================
    // ZOOM KỸ THUẬT SỐ (DIGITAL ZOOM)
    // ========================================================================

    /**
     * @brief Zoom kỹ thuật số vào vùng quan tâm
     */
    cv::Mat ImageProcessor::digitalZoom(const cv::Mat& src,       // Ảnh gốc (BGR)
                                         float zoom_factor,       // Hệ số zoom (>1.0)
                                         cv::Point2f center)      // Tâm zoom
    {
        // Kiểm tra ảnh đầu vào
        if (src.empty()) { // Ảnh rỗng
            std::cerr << "[ImageProcessor] Loi: Anh dau vao rong cho digitalZoom." << std::endl;
            return src; // Trả về ảnh rỗng
        }

        // Giới hạn hệ số zoom trong khoảng hợp lệ [1.0, 10.0]
        zoom_factor = std::max(1.0f, std::min(10.0f, zoom_factor)); // Clamp zoom factor
        // zoom < 1 không có ý nghĩa (thu nhỏ), zoom > 10 quá mờ

        // Xác định tâm zoom
        // Nếu center = (-1, -1) tức là chưa chỉ định → dùng tâm ảnh
        if (center.x < 0 || center.y < 0) { // Tâm chưa được chỉ định
            center.x = static_cast<float>(src.cols) / 2.0f; // Tâm x = nửa chiều rộng
            center.y = static_cast<float>(src.rows) / 2.0f; // Tâm y = nửa chiều cao
        }

        // Đảm bảo tâm zoom nằm trong biên giới ảnh
        center.x = std::max(0.0f, std::min(center.x, static_cast<float>(src.cols - 1)));
        // Clamp x vào [0, cols-1]
        center.y = std::max(0.0f, std::min(center.y, static_cast<float>(src.rows - 1)));
        // Clamp y vào [0, rows-1]

        // Tính kích thước vùng cắt = kích thước gốc / zoom_factor
        // Ví dụ: ảnh 1920x1080, zoom 2x → cắt vùng 960x540 rồi phóng to lại 1920x1080
        float crop_width = static_cast<float>(src.cols) / zoom_factor;   // Chiều rộng vùng cắt
        float crop_height = static_cast<float>(src.rows) / zoom_factor;  // Chiều cao vùng cắt

        // Sử dụng getRectSubPix để cắt vùng ảnh quanh tâm zoom
        // Ưu điểm: xử lý tốt trường hợp vùng cắt vượt biên ảnh (padding tự động)
        cv::Mat cropped; // Biến chứa vùng ảnh đã cắt
        cv::getRectSubPix(
            src,                                                         // Ảnh nguồn
            cv::Size(static_cast<int>(crop_width),                       // Chiều rộng vùng cắt
                     static_cast<int>(crop_height)),                     // Chiều cao vùng cắt
            center,                                                      // Tâm vùng cắt
            cropped                                                      // Ảnh con đã cắt (output)
        );
        // getRectSubPix cắt vùng chữ nhật với tâm ở center
        // Hỗ trợ sub-pixel accuracy và tự xử lý biên (extrapolation)

        // Phóng to vùng cắt về kích thước gốc bằng nội suy tuyến tính
        cv::Mat zoomed; // Biến chứa ảnh đã zoom
        cv::resize(cropped,             // Ảnh con cần phóng to
                   zoomed,              // Ảnh đã phóng to (output)
                   src.size(),          // Kích thước đích = kích thước ảnh gốc
                   0,                   // fx = 0 (không dùng, vì đã chỉ định dsize)
                   0,                   // fy = 0 (không dùng)
                   cv::INTER_LINEAR);   // Phương pháp nội suy tuyến tính (cân bằng chất lượng/tốc độ)
        // INTER_LINEAR: nội suy song tuyến tính, tốt cho phóng to vừa phải
        // INTER_CUBIC chất lượng hơn nhưng chậm hơn, INTER_NEAREST nhanh nhưng pixel hóa

        return zoomed; // Trả về ảnh đã zoom kỹ thuật số
    }

    // ========================================================================
    // CÁC HÀM NỘI BỘ PRIVATE
    // ========================================================================

    /**
     * @brief Tính Dark Channel của ảnh (bước đầu tiên trong khử sương mù)
     */
    cv::Mat ImageProcessor::computeDarkChannel(const cv::Mat& src,     // Ảnh BGR đầu vào
                                                int patch_size)        // Kích thước vùng lân cận
    {
        // Đảm bảo patch_size là số lẻ (yêu cầu cho erode)
        if (patch_size % 2 == 0) { // Nếu là số chẵn
            patch_size += 1; // Tăng thêm 1 thành số lẻ
        }

        // Bước 1: Tách 3 kênh BGR
        std::vector<cv::Mat> channels; // Vector chứa 3 kênh B, G, R
        cv::split(src, channels); // Tách: channels[0]=B, [1]=G, [2]=R

        // Bước 2: Tính giá trị nhỏ nhất (min) trên 3 kênh tại mỗi pixel
        // Dark channel pixel-wise: dc(x) = min(B(x), G(x), R(x))
        cv::Mat min_channel; // Biến chứa giá trị min tại mỗi pixel

        cv::min(channels[0], channels[1], min_channel); // min_channel = min(B, G) element-wise
        cv::min(min_channel, channels[2], min_channel); // min_channel = min(min(B,G), R) = min(B,G,R)

        // Bước 3: Áp dụng min filter (erosion) trên vùng lân cận patch_size x patch_size
        // Erosion với kernel đều = min filter: tìm giá trị nhỏ nhất trong vùng lân cận
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT,                              // Kernel hình chữ nhật cho min filter
            cv::Size(patch_size, patch_size)              // Kích thước vùng lân cận (15x15 mặc định)
        );

        cv::Mat dark_channel; // Biến chứa dark channel cuối cùng
        cv::erode(min_channel, dark_channel, kernel); // Erosion = min filter trên vùng lân cận
        // Sau bước này: dark_channel(x) = min_{y ∈ Ω(x)} min_{c ∈ {B,G,R}} I_c(y)
        // Ω(x) là vùng patch_size x patch_size quanh pixel x

        return dark_channel; // Trả về dark channel (grayscale, 8-bit)
    }

    /**
     * @brief Ước lượng ánh sáng khí quyển từ ảnh sương mù
     */
    cv::Scalar ImageProcessor::estimateAtmosphericLight(const cv::Mat& src,             // Ảnh BGR gốc
                                                         const cv::Mat& dark_channel)   // Dark channel
    {
        // Bước 1: Tìm top 0.1% pixel sáng nhất trong dark channel
        // Vùng sáng nhất trong dark channel tương ứng với vùng sương mù dày nhất
        int num_pixels = dark_channel.rows * dark_channel.cols; // Tổng số pixel
        int top_pixels = std::max(static_cast<int>(num_pixels * 0.001), 1); // 0.1% pixel, ít nhất 1
        // 0.1% pixel sáng nhất trong dark channel = vùng có sương mù nhiều nhất

        // Bước 2: Chuyển dark channel thành vector 1D để sắp xếp
        cv::Mat dark_flat = dark_channel.reshape(1, 1).clone(); // Reshape thành 1 hàng (vector 1D)
        // reshape(channels=1, rows=1) = flatten thành 1 hàng

        // Tạo mảng index để theo dõi vị trí gốc sau khi sắp xếp
        std::vector<int> indices(num_pixels); // Vector index từ 0 đến num_pixels-1
        std::iota(indices.begin(), indices.end(), 0); // Điền: 0, 1, 2, ..., num_pixels-1
        // iota (từ <numeric>) tự động tăng dần từ giá trị khởi đầu

        // Con trỏ đến dữ liệu dark channel để truy cập nhanh
        const uchar* dark_data = dark_flat.ptr<uchar>(0); // Con trỏ đến hàng 0 (hàng duy nhất)

        // Sắp xếp index theo giá trị dark channel giảm dần (sáng nhất trước)
        std::partial_sort(indices.begin(),                           // Bắt đầu sắp xếp
                          indices.begin() + top_pixels,              // Chỉ sắp xếp top_pixels phần tử đầu
                          indices.end(),                             // Kết thúc range
                          [dark_data](int a, int b) {                // Lambda so sánh giảm dần
                              return dark_data[a] > dark_data[b];    // Pixel sáng hơn đứng trước
                          });
        // partial_sort: chỉ sắp xếp top_pixels phần tử đầu tiên, nhanh hơn sort toàn bộ

        // Bước 3: Lấy giá trị trung bình BGR tại các pixel sáng nhất trong dark channel
        double sum_b = 0.0, sum_g = 0.0, sum_r = 0.0; // Tổng giá trị B, G, R

        for (int i = 0; i < top_pixels; ++i) { // Duyệt qua top pixel sáng nhất
            int idx = indices[i]; // Index pixel trong dark channel
            int row = idx / dark_channel.cols; // Chuyển index → tọa độ hàng
            int col = idx % dark_channel.cols; // Chuyển index → tọa độ cột

            // Lấy giá trị BGR tại vị trí (row, col) trên ảnh gốc
            cv::Vec3b pixel = src.at<cv::Vec3b>(row, col); // Đọc pixel BGR
            sum_b += pixel[0]; // Cộng dồn giá trị Blue
            sum_g += pixel[1]; // Cộng dồn giá trị Green
            sum_r += pixel[2]; // Cộng dồn giá trị Red
        }

        // Tính trung bình để có giá trị atmospheric light
        double a_b = sum_b / top_pixels; // Trung bình Blue
        double a_g = sum_g / top_pixels; // Trung bình Green
        double a_r = sum_r / top_pixels; // Trung bình Red

        return cv::Scalar(a_b, a_g, a_r); // Trả về ánh sáng khí quyển dạng Scalar(B, G, R)
    }

} // namespace drone_vision - Kết thúc implementation của namespace drone_vision
