# ==============================================================================
# DroneVisionPro - Hệ thống Xử lý Ảnh Chuyên nghiệp cho Drone
# ==============================================================================
# File: CMakeLists.txt
# Mô tả: File cấu hình CMake để build toàn bộ dự án
# Yêu cầu: CMake >= 3.15, OpenCV >= 4.0
# ==============================================================================

# --- Yêu cầu phiên bản CMake tối thiểu ---
cmake_minimum_required(VERSION 3.15)

# --- Khai báo tên dự án và ngôn ngữ sử dụng ---
project(DroneVisionPro 
    VERSION 1.0.0          # Phiên bản dự án
    LANGUAGES CXX          # Ngôn ngữ: C++
    DESCRIPTION "Professional Drone Image Processing System"
)

# --- Thiết lập chuẩn C++17 ---
set(CMAKE_CXX_STANDARD 17)             # Sử dụng C++17 cho các tính năng hiện đại
set(CMAKE_CXX_STANDARD_REQUIRED ON)    # Bắt buộc phải hỗ trợ C++17
set(CMAKE_CXX_EXTENSIONS OFF)          # Tắt các extension riêng của compiler

# --- Tìm và nạp thư viện OpenCV ---
# Cấu hình đường dẫn tĩnh đến thư mục OpenCV (bản MinGW) vừa tải
set(OpenCV_DIR "C:/24021238TranNgocBao/COMPUTER VISION/tools/OpenCV-MinGW-Build-OpenCV-4.5.2-x64/x64/mingw/lib")

# find_package sẽ tìm cấu hình OpenCV tại thư mục trên
find_package(OpenCV REQUIRED)

# --- In thông tin OpenCV đã tìm được (để debug) ---
message(STATUS "================================================")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")          # In phiên bản OpenCV
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}") # In đường dẫn header
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")            # In danh sách thư viện
message(STATUS "================================================")

# --- Thêm đường dẫn header files của dự án ---
# Tất cả file .hpp trong thư mục include/ sẽ được tìm thấy
include_directories(
    ${CMAKE_SOURCE_DIR}/include    # Header files của dự án
    ${OpenCV_INCLUDE_DIRS}         # Header files của OpenCV
)

# --- Liệt kê tất cả source files (.cpp) ---
set(SOURCES
    src/main.cpp                   # Entry point - hàm main và menu chương trình
    src/utils.cpp                  # Tiện ích chung: FPS, logging, drawing helpers
    src/aruco_detector.cpp         # Module phát hiện ArUco marker
    src/qr_detector.cpp            # Module phát hiện và giải mã QR code
    src/object_detector.cpp        # Module phát hiện đối tượng (YOLO DNN)
    src/image_processor.cpp        # Module xử lý ảnh (filters, dehazing, stabilize)
    src/drone_hud.cpp              # Module HUD overlay cho drone
)

# --- Liệt kê tất cả header files (.hpp) ---
set(HEADERS
    include/utils.hpp              # Header tiện ích
    include/aruco_detector.hpp     # Header ArUco detector
    include/qr_detector.hpp        # Header QR detector
    include/object_detector.hpp    # Header Object detector
    include/image_processor.hpp    # Header Image processor
    include/drone_hud.hpp          # Header Drone HUD
)

# --- Tạo executable từ source files ---
# add_executable tạo file chạy được từ tất cả source files
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# --- Link thư viện OpenCV vào executable ---
# target_link_libraries kết nối các thư viện OpenCV cần thiết
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# --- Thiết lập thư mục include cho target ---
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include    # Đường dẫn header của dự án
)

# --- Tạo thư mục output cho assets ---
# Tạo thư mục assets/test_images và assets/models nếu chưa có
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/assets/test_images)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/assets/models)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/assets/generated_markers)

# --- Copy assets sang build directory (nếu có) ---
# Để chương trình có thể tìm thấy assets khi chạy từ build directory
if(EXISTS ${CMAKE_SOURCE_DIR}/assets)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
        COMMENT "Copying assets to build directory..."   # Thông báo khi copy
    )
endif()

# --- Thiết lập cho Windows ---
if(WIN32)
    # Trên Windows, thêm flag để console hiển thị Unicode
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS    # Tắt warning về hàm C không an toàn
        NOMINMAX                    # Tránh xung đột min/max macro của Windows
    )
endif()

# --- In hướng dẫn build ---
message(STATUS "")
message(STATUS "=== HUONG DAN BUILD ===")
message(STATUS "1. mkdir build && cd build")
message(STATUS "2. cmake ..")
message(STATUS "3. cmake --build . --config Release")
message(STATUS "4. Run: ./DroneVisionPro (hoac DroneVisionPro.exe)")
message(STATUS "========================")
