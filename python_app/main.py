import cv2
import cv2.aruco as aruco
import numpy as np
import time
from pyzbar.pyzbar import decode

def draw_hud(frame, fps, altitude, detected_count):
    h, w = frame.shape[:2]
    # Vẽ Crosshair (Tâm ngắm)
    cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 2)
    cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 2)
    
    # Vẽ thanh Status bar dưới cùng
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"ALT: {altitude:.1f}m", (w - 120, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"TARGETS: {detected_count}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

def main():
    print("[INFO] Khoi dong he thong DroneVisionPro (Phien ban Python)...")
    
    # Khoi tao Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Khong the mo camera!")
        return
        
    # Khoi tao thong so Camera gia lap de do khoang cach bang met (m)
    cam_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cam_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    camera_matrix = np.array([[cam_w, 0, cam_w/2], [0, cam_w, cam_h/2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))
    marker_length_m = 0.20 # Kich thuoc thuc te cua bai dap (0.2m = 20cm)
        
    # Khoi tao YOLOv4-tiny
    print("[INFO] Tai mo hinh YOLO...")
    net = cv2.dnn.readNetFromDarknet("../assets/models/yolov4-tiny.cfg", "../assets/models/yolov4-tiny.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    with open("../assets/models/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Khoi tao ArUco (Ho tro OpenCV 4.7+)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX # Tang do chinh xac dang ke
    detector = aruco.ArucoDetector(dictionary, parameters)
    
    # Khoi tao QR Code (Da nang cap len PyZbar sieu nhay)
    # Khong can khai bao qr_detector nua
    
    # Bien tinh FPS
    prev_time = time.time()
    altitude = 10.5
    
    # Bien cho YOLO Frame Skipping (giup camera khong bi giat lag)
    frame_count = 0
    yolo_boxes = []
    yolo_confidences = []
    yolo_class_ids = []
    
    print("[INFO] He thong san sang! Nhan 'q' de thoat.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        detected_count = 0
        h, w = frame.shape[:2]
        
        # Tien xu ly anh (Tang tuong phan) de chong loa khi quet man hinh dien thoai
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray_frame)
        
        # 1. Phat hien ArUco (Su dung anh da chong loa)
        corners, ids, rejected = detector.detectMarkers(enhanced_gray)
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            detected_count += len(ids)
            for i in range(len(ids)):
                # Tinh tam marker
                c = corners[i][0]
                cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
                
                # Tinh toan khoang cach tu Camera toi vat the bang met (m)
                obj_points = np.array([[-marker_length_m/2, marker_length_m/2, 0],
                                       [marker_length_m/2, marker_length_m/2, 0],
                                       [marker_length_m/2, -marker_length_m/2, 0],
                                       [-marker_length_m/2, -marker_length_m/2, 0]], dtype=np.float32)
                _, rvec, tvec = cv2.solvePnP(obj_points, c, camera_matrix, dist_coeffs)
                distance_m = np.linalg.norm(tvec)
                
                if ids[i][0] == 0:
                    cv2.putText(frame, f"LANDING PAD - Cach: {distance_m:.2f} m", (cx-60, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 2)
                    
        # 2. Phat hien QR Code (Nang cap len PyZbar, doc cuc nhay va xa)
        # Dung luon anh xam da tang tuong phan (enhanced_gray) de QR cung nhay hon
        decoded_objects = decode(enhanced_gray)
        for obj in decoded_objects:
            info = obj.data.decode('utf-8')
            pts = np.array([obj.polygon], np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
            
            x, y = obj.rect.left, obj.rect.top
            cv2.putText(frame, f"QR: {info}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            detected_count += 1
                    
        # 3. Phat hien YOLO (Object Detection)
        # Toi uu hoa: Chi chay YOLO moi 3 frame de Camera hien thi muot ma hon (chong giat lag)
        frame_count += 1
        if frame_count % 3 == 0:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            yolo_class_ids = []
            yolo_confidences = []
            yolo_boxes = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        width = int(detection[2] * w)
                        height = int(detection[3] * h)
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)
                        yolo_boxes.append([x, y, width, height])
                        yolo_confidences.append(float(confidence))
                        yolo_class_ids.append(class_id)
                        
        # Ve hinh chu nhat cua vat the (su dung du lieu tu cac frame truoc de luon hien thi muot)
        indices = cv2.dnn.NMSBoxes(yolo_boxes, yolo_confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w_box, h_box = yolo_boxes[i]
                label = str(classes[yolo_class_ids[i]])
                conf = yolo_confidences[i]
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 165, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                detected_count += 1
                
        # 4. Tinh FPS & Ve HUD
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        draw_hud(frame, fps, altitude, detected_count)
        
        # Hien thi anh
        cv2.imshow("DroneVisionPro (Python)", frame)
        
        # Kiem tra xem nguoi dung co bam nut 'X' de dong cua so khong (Cach an toan hon tren Windows)
        if cv2.getWindowProperty("DroneVisionPro (Python)", cv2.WND_PROP_AUTOSIZE) < 0:
            break
            
        # Nhan 'q' de thoat
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Giai phong camera va dong cua so an toan
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
