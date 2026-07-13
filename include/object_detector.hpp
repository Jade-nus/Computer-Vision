/**
 * @file object_detector.hpp
 * @brief Lớp phát hiện đối tượng sử dụng mô hình YOLO thông qua OpenCV DNN
 * @details Module này cung cấp khả năng phát hiện đối tượng thời gian thực
 *          cho ứng dụng drone sử dụng mạng neural YOLO (You Only Look Once)
 * @author Tran Ngoc Bao - 24021238
 * @date 2026
 */

#pragma once // Đảm bảo file header chỉ được include một lần duy nhất, tránh lỗi định nghĩa trùng lặp

// === CÁC THƯ VIỆN CHUẨN C++ ===
#include <string>          // Thư viện xử lý chuỗi ký tự std::string
#include <vector>          // Thư viện container động std::vector cho danh sách phát hiện
#include <map>             // Thư viện container std::map cho việc đếm đối tượng theo lớp
#include <fstream>         // Thư viện đọc/ghi file để load danh sách tên lớp
#include <random>          // Thư viện sinh số ngẫu nhiên để tạo màu sắc cho bounding box

// === CÁC THƯ VIỆN OPENCV ===
#include <opencv2/opencv.hpp>      // Thư viện OpenCV chính với tất cả module cơ bản
#include <opencv2/dnn.hpp>         // Module Deep Neural Network của OpenCV để chạy mô hình YOLO
#include <opencv2/dnn/all_layers.hpp> // Tất cả các layer DNN hỗ trợ trong OpenCV

/**
 * @namespace drone_vision
 * @brief Không gian tên chứa toàn bộ các module xử lý thị giác máy tính cho drone
 * @details Tổ chức code theo namespace giúp tránh xung đột tên và dễ quản lý
 */
namespace drone_vision {

    /**
     * @struct Detection
     * @brief Cấu trúc lưu trữ thông tin một đối tượng được phát hiện
     * @details Mỗi Detection đại diện cho một bounding box với thông tin lớp,
     *          độ tin cậy, vị trí và màu sắc hiển thị
     */
    struct Detection {
        int classId;              // ID của lớp đối tượng (0, 1, 2,...) tương ứng với file .names
        std::string className;    // Tên lớp đối tượng dạng chuỗi (vd: "person", "car", "drone")
        float confidence;         // Độ tin cậy của phát hiện, giá trị từ 0.0 đến 1.0 (1.0 = chắc chắn 100%)
        cv::Rect bbox;            // Hình chữ nhật bao quanh đối tượng (x, y, width, height) tính bằng pixel
        cv::Scalar color;         // Màu sắc BGR để vẽ bounding box, mỗi lớp có một màu riêng biệt
    };

    /**
     * @class ObjectDetector
     * @brief Lớp phát hiện đối tượng sử dụng mô hình YOLO qua OpenCV DNN
     * @details Lớp này đóng gói toàn bộ quy trình phát hiện đối tượng:
     *          1. Tải mô hình YOLO (config + weights + class names)
     *          2. Tiền xử lý ảnh đầu vào thành blob
     *          3. Chạy forward pass qua mạng neural
     *          4. Hậu xử lý: lọc theo confidence, áp dụng NMS
     *          5. Vẽ kết quả lên khung hình
     * 
     * Ví dụ sử dụng:
     * @code
     *   drone_vision::ObjectDetector detector("yolov4.cfg", "yolov4.weights", "coco.names");
     *   detector.detect(frame, 0.5f, 0.4f);
     *   detector.drawDetections(frame);
     * @endcode
     */
    class ObjectDetector {

    public:
        /**
         * @brief Constructor mặc định - khởi tạo detector ở trạng thái chưa tải mô hình
         * @details Tạo một ObjectDetector rỗng, cần gọi loadModel() trước khi sử dụng
         */
        ObjectDetector(); // Constructor mặc định không tham số

        /**
         * @brief Constructor có tham số - khởi tạo và tải mô hình YOLO ngay lập tức
         * @param cfg_path Đường dẫn tới file cấu hình YOLO (.cfg) chứa kiến trúc mạng
         * @param weights_path Đường dẫn tới file trọng số YOLO (.weights) chứa tham số đã huấn luyện
         * @param names_path Đường dẫn tới file danh sách tên lớp (.names) mỗi dòng một tên lớp
         */
        ObjectDetector(const std::string& cfg_path,          // Đường dẫn file cấu hình mạng YOLO
                       const std::string& weights_path,      // Đường dẫn file trọng số đã huấn luyện
                       const std::string& names_path);       // Đường dẫn file chứa tên các lớp đối tượng

        /**
         * @brief Destructor - giải phóng tài nguyên mạng neural và bộ nhớ
         */
        ~ObjectDetector(); // Hủy đối tượng và giải phóng tài nguyên DNN

        /**
         * @brief Tải mô hình YOLO từ các file cấu hình, trọng số và tên lớp
         * @param cfg_path Đường dẫn file .cfg chứa cấu trúc mạng (số layer, filter, stride,...)
         * @param weights_path Đường dẫn file .weights chứa trọng số đã huấn luyện trên dataset
         * @param names_path Đường dẫn file .names liệt kê tên các lớp đối tượng
         * @return true nếu tải thành công, false nếu có lỗi (file không tồn tại, format sai,...)
         */
        bool loadModel(const std::string& cfg_path,          // File cấu hình kiến trúc mạng neural
                       const std::string& weights_path,      // File trọng số đã được huấn luyện sẵn
                       const std::string& names_path);       // File danh sách tên lớp đối tượng

        /**
         * @brief Phát hiện tất cả đối tượng trong khung hình
         * @param frame Ảnh đầu vào dạng cv::Mat (BGR, 8-bit, bất kỳ kích thước)
         * @param conf_threshold Ngưỡng độ tin cậy tối thiểu (0.0-1.0), mặc định 0.5
         *        Các phát hiện có confidence < ngưỡng này sẽ bị loại bỏ
         * @param nms_threshold Ngưỡng Non-Maximum Suppression (0.0-1.0), mặc định 0.4
         *        IoU > ngưỡng này giữa 2 box thì box có confidence thấp hơn bị loại
         * @return Vector chứa tất cả các Detection đã qua lọc NMS
         */
        std::vector<Detection> detect(const cv::Mat& frame,              // Khung hình cần phát hiện đối tượng
                                      float conf_threshold = 0.5f,       // Ngưỡng confidence tối thiểu để chấp nhận
                                      float nms_threshold = 0.4f);       // Ngưỡng NMS để loại bỏ box trùng lặp

        /**
         * @brief Phát hiện đối tượng chỉ trong vùng quan tâm (ROI) của khung hình
         * @param frame Ảnh đầu vào đầy đủ
         * @param roi Hình chữ nhật xác định vùng quan tâm trong ảnh
         * @details Cắt vùng ROI, chạy detect trên vùng đó, rồi chuyển tọa độ bbox
         *          về hệ tọa độ gốc của ảnh đầy đủ
         * @return Vector chứa các Detection với bbox đã được điều chỉnh tọa độ
         */
        std::vector<Detection> detectInROI(const cv::Mat& frame,    // Ảnh đầu vào gốc đầy đủ
                                           cv::Rect roi);           // Vùng quan tâm cần phát hiện

        /**
         * @brief Vẽ tất cả bounding box, nhãn lớp và độ tin cậy lên khung hình
         * @param frame Ảnh sẽ được vẽ trực tiếp lên (tham chiếu, sẽ bị thay đổi)
         * @details Vẽ hình chữ nhật màu bao quanh đối tượng, kèm nhãn có nền
         *          để dễ đọc trên mọi loại ảnh nền
         */
        void drawDetections(cv::Mat& frame); // Vẽ kết quả phát hiện lên ảnh gốc

        /**
         * @brief Lấy danh sách tất cả đối tượng đã phát hiện trong lần detect gần nhất
         * @return Vector chứa các Detection, rỗng nếu chưa gọi detect()
         */
        std::vector<Detection> getDetections() const; // Trả về danh sách detection hiện tại

        /**
         * @brief Đếm số lượng đối tượng được phát hiện theo từng lớp
         * @return Map với key là tên lớp (string), value là số lượng (int)
         * @details Ví dụ: {"person": 3, "car": 2, "bicycle": 1}
         */
        std::map<std::string, int> countByClass() const; // Đếm đối tượng theo tên lớp

        /**
         * @brief Kiểm tra xem mô hình YOLO đã được tải thành công hay chưa
         * @return true nếu mô hình đã sẵn sàng sử dụng, false nếu chưa tải
         */
        bool isModelLoaded() const; // Kiểm tra trạng thái tải mô hình

    private:
        // === THÀNH VIÊN PRIVATE - DỮ LIỆU NỘI BỘ CỦA LỚP ===

        cv::dnn::Net network_;                          // Mạng neural YOLO đã tải, dùng cho forward pass
        std::vector<std::string> class_names_;          // Danh sách tên tất cả các lớp đối tượng (vd: "person", "car")
        std::vector<cv::Scalar> class_colors_;          // Danh sách màu BGR tương ứng với mỗi lớp để vẽ box
        std::vector<Detection> current_detections_;     // Kết quả phát hiện mới nhất từ lần gọi detect() cuối cùng
        std::vector<std::string> output_layer_names_;   // Tên các output layer của mạng YOLO (vd: yolo_82, yolo_94, yolo_106)
        bool model_loaded_;                             // Cờ cho biết mô hình đã tải thành công hay chưa

        // Kích thước ảnh đầu vào cho mạng YOLO (thường là 416x416 hoặc 608x608)
        int input_width_;     // Chiều rộng blob đầu vào (pixel), ảnh sẽ được resize về kích thước này
        int input_height_;    // Chiều cao blob đầu vào (pixel), ảnh sẽ được resize về kích thước này

        // === PHƯƠNG THỨC PRIVATE - HÀM NỘI BỘ ===

        /**
         * @brief Tải danh sách tên lớp đối tượng từ file .names
         * @param names_path Đường dẫn tới file .names (mỗi dòng một tên lớp)
         * @return true nếu đọc thành công, false nếu file không tồn tại hoặc rỗng
         */
        bool loadClassNames(const std::string& names_path); // Đọc file tên lớp vào class_names_

        /**
         * @brief Lấy tên các output layer từ mạng YOLO đã tải
         * @return Vector chứa tên các output layer (layer cuối cùng tạo ra detection)
         * @details YOLO có 3 output layer ở 3 scale khác nhau để detect đối tượng đa kích thước
         */
        std::vector<std::string> getOutputLayerNames(); // Xác định các layer đầu ra của mạng

        /**
         * @brief Sinh màu ngẫu nhiên cho mỗi lớp đối tượng
         * @details Tạo màu BGR ngẫu nhiên sáng và dễ phân biệt cho từng lớp
         *          Sử dụng seed cố định để màu nhất quán giữa các lần chạy
         */
        void generateColors(); // Tạo bảng màu ngẫu nhiên cho các lớp

        /**
         * @brief Hậu xử lý đầu ra của mạng YOLO
         * @param frame Ảnh gốc (cần kích thước để chuyển đổi tọa độ tương đối sang tuyệt đối)
         * @param outputs Đầu ra thô từ mạng neural (ma trận chứa box, confidence, class scores)
         * @param conf_threshold Ngưỡng confidence để lọc detection
         * @param nms_threshold Ngưỡng NMS để loại box trùng
         * @details Quy trình: parse output → lọc confidence → NMS → tạo Detection objects
         */
        void postProcess(const cv::Mat& frame,                         // Ảnh gốc để lấy kích thước
                         const std::vector<cv::Mat>& outputs,          // Đầu ra thô từ forward pass
                         float conf_threshold,                         // Ngưỡng confidence
                         float nms_threshold);                         // Ngưỡng NMS
    };

} // namespace drone_vision - Kết thúc không gian tên drone_vision
