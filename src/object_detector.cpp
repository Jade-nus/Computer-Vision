/**
 * @file object_detector.cpp
 * @brief Triển khai đầy đủ lớp ObjectDetector sử dụng YOLO qua OpenCV DNN
 * @details File này chứa toàn bộ implementation của class ObjectDetector:
 *          - Tải mô hình YOLO (Darknet format)
 *          - Tiền xử lý ảnh thành blob 4D
 *          - Forward pass qua mạng neural
 *          - Hậu xử lý: parse output, lọc confidence, NMS
 *          - Vẽ kết quả phát hiện lên khung hình
 * @author Tran Ngoc Bao - 24021238
 * @date 2026
 */

#include "object_detector.hpp" // Include header chứa khai báo lớp ObjectDetector và struct Detection

// Include thêm thư viện chuẩn cần thiết cho implementation
#include <iostream>   // Thư viện xuất/nhập console để in thông báo lỗi và debug
#include <sstream>    // Thư viện xử lý chuỗi dòng để format text hiển thị
#include <numeric>    // Thư viện tính toán số học: iota, accumulate,...

/**
 * @namespace drone_vision
 * @brief Không gian tên chứa toàn bộ module thị giác cho drone
 */
namespace drone_vision {

    // ========================================================================
    // CONSTRUCTOR VÀ DESTRUCTOR
    // ========================================================================

    /**
     * @brief Constructor mặc định - tạo detector ở trạng thái chưa sẵn sàng
     */
    ObjectDetector::ObjectDetector()
        : model_loaded_(false),       // Đặt cờ tải mô hình = false vì chưa tải gì
          input_width_(416),          // Kích thước chiều rộng đầu vào mặc định 416 pixel (chuẩn YOLOv3/v4)
          input_height_(416)          // Kích thước chiều cao đầu vào mặc định 416 pixel (phải là bội số của 32)
    {
        // Constructor rỗng - không làm gì thêm
        // Người dùng cần gọi loadModel() trước khi sử dụng detect()
    }

    /**
     * @brief Constructor có tham số - tải mô hình YOLO ngay khi khởi tạo
     */
    ObjectDetector::ObjectDetector(const std::string& cfg_path,       // Đường dẫn file cấu hình .cfg
                                   const std::string& weights_path,   // Đường dẫn file trọng số .weights
                                   const std::string& names_path)     // Đường dẫn file tên lớp .names
        : model_loaded_(false),      // Khởi tạo cờ = false, sẽ đặt true nếu tải thành công
          input_width_(416),         // Kích thước đầu vào mặc định 416x416
          input_height_(416)         // YOLO yêu cầu ảnh vuông với kích thước là bội của 32
    {
        // Gọi loadModel() để tải mô hình ngay trong constructor
        loadModel(cfg_path, weights_path, names_path); // Tải mô hình, kết quả lưu vào model_loaded_
    }

    /**
     * @brief Destructor - giải phóng tài nguyên
     */
    ObjectDetector::~ObjectDetector()
    {
        // OpenCV tự quản lý bộ nhớ DNN Net thông qua smart pointer nội bộ
        // Không cần giải phóng thủ công, nhưng xóa danh sách detection để đảm bảo sạch sẽ
        current_detections_.clear(); // Xóa danh sách detection hiện tại
        class_names_.clear();        // Xóa danh sách tên lớp
        class_colors_.clear();       // Xóa danh sách màu sắc
    }

    // ========================================================================
    // TẢI MÔ HÌNH YOLO
    // ========================================================================

    /**
     * @brief Tải mô hình YOLO từ file Darknet (.cfg + .weights) và file tên lớp (.names)
     */
    bool ObjectDetector::loadModel(const std::string& cfg_path,       // File .cfg chứa kiến trúc mạng
                                   const std::string& weights_path,   // File .weights chứa trọng số
                                   const std::string& names_path)     // File .names chứa tên lớp
    {
        try {
            // Bước 1: Tải danh sách tên lớp đối tượng từ file .names
            if (!loadClassNames(names_path)) {   // Gọi hàm đọc file tên lớp
                // In thông báo lỗi nếu không đọc được file tên lớp
                std::cerr << "[ObjectDetector] Loi: Khong the doc file ten lop: " << names_path << std::endl;
                return false; // Trả về false cho biết tải thất bại
            }

            // Bước 2: Tải mạng neural từ file Darknet
            // readNetFromDarknet đọc cấu trúc mạng từ .cfg và trọng số từ .weights
            network_ = cv::dnn::readNetFromDarknet(cfg_path, weights_path); // Tạo mạng DNN từ Darknet format

            // Kiểm tra xem mạng có được tải thành công hay không
            if (network_.empty()) { // Mạng rỗng = tải thất bại
                // In thông báo lỗi chi tiết với đường dẫn file
                std::cerr << "[ObjectDetector] Loi: Khong the tai mo hinh YOLO tu: "
                          << cfg_path << " va " << weights_path << std::endl;
                return false; // Trả về false cho biết tải thất bại
            }

            // Bước 3: Thiết lập backend tính toán cho mạng neural
            // OPENCV backend sử dụng OpenCV tự có, tương thích mọi nền tảng
            network_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); // Sử dụng backend OpenCV (không cần CUDA/Intel)

            // Bước 4: Thiết lập target thiết bị tính toán
            // CPU target chạy trên CPU, phù hợp cho hầu hết hệ thống nhúng drone
            network_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // Chạy inference trên CPU (thay vì GPU)

            // Bước 5: Lấy tên các output layer từ mạng YOLO
            // YOLO có nhiều output layer ở các scale khác nhau (3 cho YOLOv3/v4)
            output_layer_names_ = getOutputLayerNames(); // Xác định output layers để lấy kết quả detection

            // Bước 6: Sinh màu ngẫu nhiên cho mỗi lớp đối tượng
            generateColors(); // Tạo bảng màu để vẽ bounding box cho từng lớp

            // Đặt cờ tải thành công
            model_loaded_ = true; // Đánh dấu mô hình đã sẵn sàng sử dụng

            // In thông báo thành công kèm số lượng lớp
            std::cout << "[ObjectDetector] Da tai mo hinh YOLO thanh cong voi "
                      << class_names_.size() << " lop doi tuong." << std::endl;

            return true; // Trả về true cho biết tải thành công

        } catch (const cv::Exception& e) {
            // Bắt exception từ OpenCV nếu có lỗi xảy ra trong quá trình tải
            std::cerr << "[ObjectDetector] Loi OpenCV: " << e.what() << std::endl;
            model_loaded_ = false; // Đặt cờ = false vì tải thất bại
            return false;          // Trả về false
        } catch (const std::exception& e) {
            // Bắt exception chuẩn C++ cho các lỗi khác
            std::cerr << "[ObjectDetector] Loi: " << e.what() << std::endl;
            model_loaded_ = false; // Đặt cờ = false
            return false;          // Trả về false
        }
    }

    // ========================================================================
    // PHÁT HIỆN ĐỐI TƯỢNG
    // ========================================================================

    /**
     * @brief Phát hiện tất cả đối tượng trong một khung hình
     */
    std::vector<Detection> ObjectDetector::detect(const cv::Mat& frame,         // Ảnh đầu vào BGR
                                                   float conf_threshold,        // Ngưỡng confidence (0.0-1.0)
                                                   float nms_threshold)         // Ngưỡng NMS (0.0-1.0)
    {
        // Kiểm tra mô hình đã được tải chưa trước khi chạy detection
        if (!model_loaded_) { // Nếu mô hình chưa tải
            // In cảnh báo cho người dùng biết cần tải mô hình trước
            std::cerr << "[ObjectDetector] Canh bao: Mo hinh chua duoc tai. Goi loadModel() truoc." << std::endl;
            return std::vector<Detection>(); // Trả về vector rỗng, không có detection nào
        }

        // Kiểm tra ảnh đầu vào có hợp lệ không
        if (frame.empty()) { // Ảnh rỗng = không có dữ liệu pixel
            // In cảnh báo về ảnh rỗng
            std::cerr << "[ObjectDetector] Canh bao: Anh dau vao rong." << std::endl;
            return std::vector<Detection>(); // Trả về vector rỗng
        }

        // Bước 1: Tạo blob 4D từ ảnh đầu vào
        // blobFromImage chuyển ảnh BGR thành tensor 4D (batch, channels, height, width)
        // Tham số: ảnh, scalefactor (1/255 để normalize về [0,1]), kích thước đầu vào,
        //          mean subtraction (0,0,0 = không trừ mean), swapRB (true = BGR→RGB), crop (false)
        cv::Mat blob = cv::dnn::blobFromImage(
            frame,                                            // Ảnh đầu vào gốc (BGR, 8-bit)
            1.0 / 255.0,                                     // Hệ số scale: chia 255 để normalize pixel về [0, 1]
            cv::Size(input_width_, input_height_),            // Resize ảnh về kích thước mạng yêu cầu (416x416)
            cv::Scalar(0, 0, 0),                              // Mean subtraction: không trừ giá trị trung bình
            true,                                             // swapRB: đổi BGR sang RGB vì YOLO huấn luyện trên RGB
            false                                             // crop: không cắt ảnh, chỉ resize (có thể bị méo)
        );

        // Bước 2: Đặt blob làm đầu vào cho mạng neural
        network_.setInput(blob); // Truyền tensor 4D vào input layer của mạng YOLO

        // Bước 3: Chạy forward pass để lấy đầu ra từ các output layers
        std::vector<cv::Mat> outputs; // Vector chứa kết quả đầu ra từ mỗi output layer
        network_.forward(outputs, output_layer_names_); // Chạy mạng và lấy đầu ra từ 3 scale layers

        // Bước 4: Hậu xử lý đầu ra - parse, lọc confidence, NMS
        postProcess(frame, outputs, conf_threshold, nms_threshold); // Xử lý output thành Detection objects

        // Trả về danh sách tất cả đối tượng đã phát hiện sau khi lọc
        return current_detections_; // Danh sách Detection đã được cập nhật bởi postProcess()
    }

    /**
     * @brief Phát hiện đối tượng chỉ trong vùng quan tâm (ROI) của khung hình
     */
    std::vector<Detection> ObjectDetector::detectInROI(const cv::Mat& frame,   // Ảnh gốc đầy đủ
                                                        cv::Rect roi)          // Vùng quan tâm (x, y, w, h)
    {
        // Kiểm tra mô hình đã tải chưa
        if (!model_loaded_) { // Nếu chưa tải mô hình
            std::cerr << "[ObjectDetector] Canh bao: Mo hinh chua duoc tai." << std::endl;
            return std::vector<Detection>(); // Trả về rỗng
        }

        // Kiểm tra ảnh đầu vào hợp lệ
        if (frame.empty()) { // Ảnh rỗng
            std::cerr << "[ObjectDetector] Canh bao: Anh dau vao rong." << std::endl;
            return std::vector<Detection>(); // Trả về rỗng
        }

        // Đảm bảo ROI nằm hoàn toàn trong biên giới ảnh
        // Toán tử & giữa Rect và Rect tạo ra giao (intersection) của 2 hình chữ nhật
        cv::Rect safe_roi = roi & cv::Rect(0, 0, frame.cols, frame.rows); // Cắt ROI để không vượt ngoài ảnh

        // Kiểm tra ROI sau khi cắt có còn hợp lệ không
        if (safe_roi.width <= 0 || safe_roi.height <= 0) { // ROI có diện tích = 0 hoặc âm
            std::cerr << "[ObjectDetector] Canh bao: ROI khong hop le hoac nam ngoai anh." << std::endl;
            return std::vector<Detection>(); // Trả về rỗng vì không có vùng nào để detect
        }

        // Cắt vùng ROI từ ảnh gốc
        // frame(safe_roi) tạo một cv::Mat con trỏ đến vùng nhớ của ảnh gốc (không copy)
        cv::Mat roi_frame = frame(safe_roi).clone(); // Clone để tạo bản sao độc lập tránh ảnh hưởng ảnh gốc

        // Chạy detection trên vùng ROI đã cắt
        std::vector<Detection> roi_detections = detect(roi_frame); // Phát hiện đối tượng trong vùng ROI

        // Chuyển đổi tọa độ bbox từ hệ tọa độ ROI sang hệ tọa độ ảnh gốc
        for (auto& det : roi_detections) { // Duyệt qua từng detection trong kết quả
            // Cộng tọa độ gốc của ROI vào tọa độ bbox để chuyển về hệ tọa độ ảnh đầy đủ
            det.bbox.x += safe_roi.x; // Dịch tọa độ x của bbox theo vị trí x của ROI
            det.bbox.y += safe_roi.y; // Dịch tọa độ y của bbox theo vị trí y của ROI
        }

        // Cập nhật danh sách detection nội bộ với kết quả từ ROI
        current_detections_ = roi_detections; // Lưu kết quả ROI detection làm detection hiện tại

        return roi_detections; // Trả về danh sách detection với tọa độ đã chuyển đổi
    }

    // ========================================================================
    // VẼ KẾT QUẢ PHÁT HIỆN
    // ========================================================================

    /**
     * @brief Vẽ tất cả bounding box và nhãn lên khung hình
     */
    void ObjectDetector::drawDetections(cv::Mat& frame) // Ảnh sẽ bị thay đổi trực tiếp (tham chiếu)
    {
        // Duyệt qua từng detection trong danh sách kết quả hiện tại
        for (const auto& det : current_detections_) { // Lặp qua mỗi đối tượng đã phát hiện

            // Vẽ hình chữ nhật bounding box bao quanh đối tượng
            cv::rectangle(frame,            // Ảnh đích để vẽ lên
                          det.bbox,         // Hình chữ nhật bao quanh (x, y, width, height)
                          det.color,        // Màu sắc riêng của lớp đối tượng này
                          2);               // Độ dày đường viền = 2 pixel (đủ rõ mà không che ảnh)

            // Tạo chuỗi nhãn hiển thị: "TenLop: 95%" (tên lớp kèm độ tin cậy phần trăm)
            std::ostringstream label_stream; // Dùng ostringstream để format chuỗi linh hoạt
            label_stream << det.className                    // Tên lớp đối tượng (vd: "person")
                         << ": "                             // Dấu phân cách giữa tên và confidence
                         << std::fixed                       // Sử dụng định dạng số thập phân cố định
                         << std::setprecision(0)             // Không hiển thị chữ số thập phân
                         << (det.confidence * 100.0f)        // Nhân 100 để chuyển từ [0,1] sang phần trăm
                         << "%";                             // Thêm ký hiệu phần trăm
            std::string label = label_stream.str(); // Chuyển ostringstream thành std::string

            // Tính kích thước text để vẽ nền (background) cho nhãn
            int baseline = 0; // Biến nhận giá trị baseline (khoảng cách từ đáy text đến baseline)
            cv::Size label_size = cv::getTextSize(
                label,                        // Chuỗi text cần đo kích thước
                cv::FONT_HERSHEY_SIMPLEX,     // Font chữ đơn giản, dễ đọc
                0.5,                          // Hệ số scale font (0.5 = cỡ vừa phải)
                1,                            // Độ dày nét chữ = 1 pixel
                &baseline                     // Con trỏ nhận giá trị baseline
            );

            // Tính vị trí y cho nhãn: đặt phía trên bounding box
            // Nếu không đủ chỗ phía trên, đặt phía dưới cạnh trên của box
            int label_y = std::max(det.bbox.y,                        // Đỉnh trên của bbox
                                   label_size.height + 10);           // Đảm bảo nhãn không bị cắt ngoài ảnh

            // Tọa độ góc trên-trái của hình chữ nhật nền cho nhãn
            cv::Point label_top_left(det.bbox.x,                                    // Căn trái theo bbox
                                     label_y - label_size.height - 10);             // Phía trên text
            // Tọa độ góc dưới-phải của hình chữ nhật nền cho nhãn
            cv::Point label_bottom_right(det.bbox.x + label_size.width + 10,        // Rộng bằng text + padding
                                         label_y);                                  // Ngang với đáy text

            // Vẽ hình chữ nhật nền đặc (filled) cho nhãn để text dễ đọc trên mọi nền
            cv::rectangle(frame,                 // Ảnh đích
                          label_top_left,        // Góc trên-trái nền nhãn
                          label_bottom_right,    // Góc dưới-phải nền nhãn
                          det.color,             // Cùng màu với bounding box
                          cv::FILLED);           // FILLED = tô đặc hình chữ nhật

            // Vẽ text nhãn lên trên nền vừa tạo, màu trắng để tương phản
            cv::putText(frame,                           // Ảnh đích
                        label,                           // Chuỗi text nhãn "ClassName: XX%"
                        cv::Point(det.bbox.x + 5,        // Vị trí x: lùi vào 5px từ mép trái
                                  label_y - 5),          // Vị trí y: lùi lên 5px từ đáy nền
                        cv::FONT_HERSHEY_SIMPLEX,        // Font chữ đơn giản, rõ ràng
                        0.5,                             // Kích thước font = 0.5
                        cv::Scalar(255, 255, 255),       // Màu trắng (BGR) cho text
                        1);                              // Độ dày nét chữ = 1 pixel
        }

        // Vẽ thông tin tổng số đối tượng phát hiện ở góc trên-trái ảnh
        std::string info_text = "Phat hien: " + std::to_string(current_detections_.size()) + " doi tuong";
        // Tạo chuỗi thông tin: "Phat hien: N doi tuong"

        // Vẽ nền đen bán trong suốt cho text thông tin
        cv::rectangle(frame,                           // Ảnh đích
                      cv::Point(0, 0),                 // Góc trên-trái (gốc ảnh)
                      cv::Point(280, 30),              // Góc dưới-phải (280x30 pixel)
                      cv::Scalar(0, 0, 0),             // Màu đen cho nền
                      cv::FILLED);                     // Tô đặc

        // Vẽ text thông tin số đối tượng lên nền đen
        cv::putText(frame,                             // Ảnh đích
                    info_text,                         // Chuỗi "Phat hien: N doi tuong"
                    cv::Point(10, 22),                 // Vị trí vẽ text (x=10, y=22)
                    cv::FONT_HERSHEY_SIMPLEX,          // Font đơn giản
                    0.6,                               // Kích thước font 0.6
                    cv::Scalar(0, 255, 0),             // Màu xanh lá (BGR) cho text
                    2);                                // Độ dày 2 pixel để nổi bật
    }

    // ========================================================================
    // CÁC HÀM TRUY VẤN (GETTER)
    // ========================================================================

    /**
     * @brief Lấy danh sách detection hiện tại
     */
    std::vector<Detection> ObjectDetector::getDetections() const
    {
        return current_detections_; // Trả về bản sao của danh sách detection gần nhất
    }

    /**
     * @brief Đếm số đối tượng theo tên lớp
     */
    std::map<std::string, int> ObjectDetector::countByClass() const
    {
        std::map<std::string, int> class_count; // Map lưu số lượng: key=tên lớp, value=số lượng

        // Duyệt qua từng detection và tăng bộ đếm tương ứng
        for (const auto& det : current_detections_) { // Lặp qua mỗi detection
            class_count[det.className]++; // Tăng bộ đếm cho lớp này (tự tạo entry mới nếu chưa có)
        }

        return class_count; // Trả về map đếm, ví dụ: {"person": 3, "car": 2}
    }

    /**
     * @brief Kiểm tra trạng thái tải mô hình
     */
    bool ObjectDetector::isModelLoaded() const
    {
        return model_loaded_; // Trả về true nếu mô hình đã tải thành công, false nếu chưa
    }

    // ========================================================================
    // CÁC HÀM NỘI BỘ (PRIVATE)
    // ========================================================================

    /**
     * @brief Tải danh sách tên lớp từ file .names
     */
    bool ObjectDetector::loadClassNames(const std::string& names_path) // Đường dẫn file .names
    {
        // Mở file .names để đọc danh sách tên lớp đối tượng
        std::ifstream file(names_path); // Tạo ifstream để đọc file text

        // Kiểm tra file có mở được không
        if (!file.is_open()) { // File không tồn tại hoặc không có quyền đọc
            std::cerr << "[ObjectDetector] Loi: Khong the mo file: " << names_path << std::endl;
            return false; // Trả về false cho biết đọc thất bại
        }

        // Xóa danh sách tên lớp cũ (nếu có) trước khi đọc mới
        class_names_.clear(); // Đảm bảo danh sách sạch trước khi thêm

        std::string line; // Biến tạm lưu từng dòng đọc từ file

        // Đọc file từng dòng, mỗi dòng là tên một lớp đối tượng
        while (std::getline(file, line)) { // Đọc đến hết file
            // Loại bỏ ký tự xuống dòng \r nếu file có format Windows (CRLF)
            if (!line.empty() && line.back() == '\r') { // Kiểm tra ký tự cuối có phải \r không
                line.pop_back(); // Xóa ký tự \r ở cuối dòng
            }

            // Chỉ thêm dòng không rỗng vào danh sách
            if (!line.empty()) { // Bỏ qua dòng trống
                class_names_.push_back(line); // Thêm tên lớp vào cuối vector
            }
        }

        file.close(); // Đóng file sau khi đọc xong

        // In thông tin số lớp đã đọc được
        std::cout << "[ObjectDetector] Da doc " << class_names_.size()
                  << " ten lop tu: " << names_path << std::endl;

        // Trả về true nếu đọc được ít nhất 1 tên lớp
        return !class_names_.empty(); // true nếu có ít nhất 1 lớp, false nếu file rỗng
    }

    /**
     * @brief Lấy tên các output layer từ mạng YOLO đã tải
     */
    std::vector<std::string> ObjectDetector::getOutputLayerNames()
    {
        // Vector lưu tên các output layer (layer cuối cùng tạo ra detection)
        std::vector<std::string> names; // Danh sách tên output layer

        // Lấy chỉ số (index) của tất cả output layer trong mạng
        // Output layer là layer không có kết nối đến layer nào phía sau
        std::vector<int> out_layers = network_.getUnconnectedOutLayers(); // Lấy index các unconnected layers

        // Lấy tên của tất cả các layer trong mạng
        std::vector<std::string> layer_names = network_.getLayerNames(); // Danh sách tên mọi layer

        // Chuyển đổi từ index sang tên layer
        for (size_t i = 0; i < out_layers.size(); ++i) { // Duyệt qua từng output layer index
            // out_layers[i] là 1-indexed trong OpenCV, nên trừ 1 để truy cập vector 0-indexed
            names.push_back(layer_names[static_cast<size_t>(out_layers[i]) - 1]); // Thêm tên layer tương ứng
        }

        return names; // Trả về danh sách tên output layers (vd: "yolo_82", "yolo_94", "yolo_106")
    }

    /**
     * @brief Sinh màu ngẫu nhiên cho mỗi lớp đối tượng
     */
    void ObjectDetector::generateColors()
    {
        class_colors_.clear(); // Xóa bảng màu cũ trước khi tạo mới

        // Sử dụng random engine với seed cố định = 42 để kết quả nhất quán giữa các lần chạy
        std::mt19937 rng(42); // Mersenne Twister RNG với seed 42 (reproducible)

        // Tạo phân phối đều cho giá trị màu từ 50 đến 255 (tránh màu quá tối)
        std::uniform_int_distribution<int> dist(50, 255); // Giá trị BGR từ 50 đến 255

        // Sinh một màu ngẫu nhiên cho mỗi lớp đối tượng
        for (size_t i = 0; i < class_names_.size(); ++i) { // Lặp qua số lượng lớp
            int b = dist(rng); // Sinh giá trị Blue ngẫu nhiên trong [50, 255]
            int g = dist(rng); // Sinh giá trị Green ngẫu nhiên trong [50, 255]
            int r = dist(rng); // Sinh giá trị Red ngẫu nhiên trong [50, 255]

            // Tạo Scalar BGR và thêm vào bảng màu
            class_colors_.push_back(cv::Scalar(b, g, r)); // Thêm màu BGR cho lớp thứ i
        }
    }

    /**
     * @brief Hậu xử lý đầu ra của mạng YOLO - trích xuất detection từ raw output
     */
    void ObjectDetector::postProcess(const cv::Mat& frame,                      // Ảnh gốc (cần kích thước)
                                      const std::vector<cv::Mat>& outputs,      // Đầu ra thô từ mạng
                                      float conf_threshold,                     // Ngưỡng confidence
                                      float nms_threshold)                      // Ngưỡng NMS
    {
        // Xóa danh sách detection cũ để chuẩn bị cho kết quả mới
        current_detections_.clear(); // Đảm bảo danh sách sạch trước khi thêm

        // Các vector tạm để lưu trữ trước khi áp dụng NMS
        std::vector<int> class_ids;          // ID lớp của mỗi detection
        std::vector<float> confidences;      // Độ tin cậy của mỗi detection
        std::vector<cv::Rect> boxes;         // Bounding box của mỗi detection

        // Lấy kích thước ảnh gốc để chuyển đổi tọa độ tương đối sang tuyệt đối
        int frame_width = frame.cols;    // Chiều rộng ảnh gốc (pixel)
        int frame_height = frame.rows;   // Chiều cao ảnh gốc (pixel)

        // Duyệt qua đầu ra của từng output layer (YOLO có 3 output layers)
        for (size_t i = 0; i < outputs.size(); ++i) { // Lặp qua 3 scale: lớn, vừa, nhỏ

            // Lấy con trỏ đến dữ liệu float của output layer hiện tại
            const float* data = reinterpret_cast<const float*>(outputs[i].data); // Con trỏ float đến data

            // Duyệt qua từng hàng (mỗi hàng = 1 detection candidate)
            // Mỗi hàng có format: [center_x, center_y, width, height, objectness, class1_score, class2_score, ...]
            for (int j = 0; j < outputs[i].rows; ++j) { // Lặp qua mỗi anchor box
                // Lấy objectness score (xác suất có đối tượng trong anchor box)
                // Vị trí index 4 chứa objectness confidence
                float objectness = data[4]; // Xác suất anchor box chứa đối tượng (0.0-1.0)

                // Chỉ xử lý nếu objectness vượt ngưỡng confidence ban đầu
                if (objectness > conf_threshold) { // Lọc sớm để giảm tính toán không cần thiết

                    // Lấy scores của tất cả các lớp (bắt đầu từ index 5)
                    // scores_ptr trỏ đến mảng các class probability scores
                    const float* scores_ptr = data + 5; // Con trỏ đến điểm số của lớp đầu tiên

                    // Tạo cv::Mat từ mảng scores để dùng cv::minMaxLoc
                    cv::Mat scores_mat(1,                                       // 1 hàng
                                       static_cast<int>(class_names_.size()),   // Số cột = số lớp
                                       CV_32FC1,                                // Kiểu float 32-bit, 1 kênh
                                       const_cast<float*>(scores_ptr));         // Dữ liệu scores

                    // Tìm lớp có score cao nhất
                    cv::Point class_id_point; // Point lưu vị trí (col = classId) của giá trị max
                    double max_class_score;   // Giá trị score cao nhất

                    cv::minMaxLoc(scores_mat,       // Ma trận scores
                                  nullptr,          // Không cần giá trị min
                                  &max_class_score, // Nhận giá trị max (score cao nhất)
                                  nullptr,          // Không cần vị trí min
                                  &class_id_point); // Nhận vị trí max (x = classId)

                    // Tính confidence cuối cùng = objectness * max_class_score
                    float final_confidence = static_cast<float>(max_class_score) * objectness;
                    // final_confidence kết hợp xác suất có đối tượng VÀ xác suất thuộc lớp cụ thể

                    // Chỉ chấp nhận detection nếu confidence cuối cùng vượt ngưỡng
                    if (final_confidence > conf_threshold) { // Lọc lần 2 với confidence chính xác hơn

                        // Trích xuất tọa độ bounding box (center_x, center_y, width, height)
                        // Giá trị từ 0 đến 1 (tương đối so với kích thước ảnh đầu vào)
                        float center_x = data[0] * frame_width;   // Tâm x tuyệt đối = tương đối * rộng ảnh
                        float center_y = data[1] * frame_height;  // Tâm y tuyệt đối = tương đối * cao ảnh
                        float box_width = data[2] * frame_width;  // Chiều rộng box tuyệt đối
                        float box_height = data[3] * frame_height; // Chiều cao box tuyệt đối

                        // Chuyển từ tọa độ tâm sang tọa độ góc trên-trái (x, y, w, h)
                        int left = static_cast<int>(center_x - box_width / 2.0f);    // x góc trên-trái
                        int top = static_cast<int>(center_y - box_height / 2.0f);     // y góc trên-trái
                        int width = static_cast<int>(box_width);                       // Chiều rộng box
                        int height = static_cast<int>(box_height);                     // Chiều cao box

                        // Lưu thông tin detection vào các vector tạm
                        class_ids.push_back(class_id_point.x);           // Lưu class ID (index của lớp)
                        confidences.push_back(final_confidence);          // Lưu confidence cuối cùng
                        boxes.push_back(cv::Rect(left, top, width, height)); // Lưu bounding box
                    }
                }

                // Di chuyển con trỏ data đến detection candidate tiếp theo
                // Mỗi row có (5 + num_classes) phần tử float
                data += outputs[i].cols; // Nhảy sang hàng tiếp theo (mỗi hàng = 1 candidate)
            }
        }

        // Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các box trùng lặp
        // NMS giữ lại box có confidence cao nhất trong nhóm box overlap
        std::vector<int> nms_indices; // Vector chứa index các box được giữ lại sau NMS

        cv::dnn::NMSBoxes(boxes,              // Danh sách tất cả bounding box
                          confidences,        // Danh sách confidence tương ứng
                          conf_threshold,     // Ngưỡng confidence (đã lọc nhưng NMS cũng cần)
                          nms_threshold,      // Ngưỡng IoU cho NMS (0.4 = loại box overlap > 40%)
                          nms_indices);       // Output: index các box sống sót qua NMS

        // Tạo danh sách Detection cuối cùng từ kết quả NMS
        for (size_t i = 0; i < nms_indices.size(); ++i) { // Duyệt qua các box đã qua NMS
            int idx = nms_indices[i]; // Lấy index gốc của box trong danh sách ban đầu

            Detection det; // Tạo struct Detection mới

            det.classId = class_ids[idx]; // Gán class ID từ kết quả phát hiện

            // Gán tên lớp, kiểm tra index hợp lệ trước khi truy cập
            if (det.classId >= 0 && det.classId < static_cast<int>(class_names_.size())) {
                det.className = class_names_[det.classId]; // Lấy tên lớp từ danh sách tên
            } else {
                det.className = "Unknown"; // Gán "Unknown" nếu classId ngoài phạm vi
            }

            det.confidence = confidences[idx]; // Gán confidence đã tính
            det.bbox = boxes[idx];             // Gán bounding box

            // Gán màu sắc tương ứng với lớp, kiểm tra index hợp lệ
            if (det.classId >= 0 && det.classId < static_cast<int>(class_colors_.size())) {
                det.color = class_colors_[det.classId]; // Lấy màu đã sinh cho lớp này
            } else {
                det.color = cv::Scalar(0, 255, 0); // Mặc định màu xanh lá nếu không có màu
            }

            current_detections_.push_back(det); // Thêm detection vào danh sách cuối cùng
        }

        // In thông tin debug: số detection trước và sau NMS
        std::cout << "[ObjectDetector] Tim thay " << boxes.size()
                  << " ung vien, giu lai " << current_detections_.size()
                  << " sau NMS." << std::endl;
    }

} // namespace drone_vision - Kết thúc implementation của namespace drone_vision
