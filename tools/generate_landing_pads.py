import cv2
import os

# --- Cấu hình ---
# Sử dụng từ điển DICT_4X4_50 như hệ thống DroneVisionPro đang dùng
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
output_dir = "landing_pads"
os.makedirs(output_dir, exist_ok=True)

# --- Tạo mã bãi đáp (Landing Pad) ---
# Mặc định Landing Pad ID là 0, tuy nhiên ta có thể tạo thêm các ID khác làm bãi dự phòng
print("Dang tao ma bai dap (Landing Pads)...")
for marker_id in range(5):
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 500)  # 500 px = kích thước ảnh
    filename = os.path.join(output_dir, f"LandingPad_ID{marker_id}.png")
    cv2.imwrite(filename, marker_img)
    print(f" -> Da tao bai dap ID {marker_id} tai {filename}")

print(f"Hoan tat! Thu muc chua bai dap: {os.path.abspath(output_dir)}")
