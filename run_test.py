import cv2
import cv2.aruco as aruco
import numpy as np
import requests
import qrcode
from pyzbar.pyzbar import decode
from PIL import Image

def main():
    print("[TEST] Bat dau tao anh gia lap...")
    # Tao anh nen trang kich thuoc 600x800
    frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
    h, w = frame.shape[:2]
    
    # 2. Add ArUco Marker (Top Left)
    dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    marker = aruco.generateImageMarker(dict_aruco, 0, 100)
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    frame[50:150, 50:150] = marker_bgr
    
    # 3. Add QR Code (Top Right)
    qr = qrcode.QRCode(version=1, box_size=3, border=1)
    qr.add_data("WAYPOINT: 10.8, 106.6")
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    qr_np = np.array(qr_img)
    qr_np = qr_np[:, :, ::-1].copy() # RGB to BGR
    qr_h, qr_w = qr_np.shape[:2]
    frame[50:150, w-150:w-50] = cv2.resize(qr_np, (100, 100))
    
    # --- BAT DAU TEST THUAT TOAN ---
    print("[TEST] Dang chay thuat toan nhan dien...")
    detected_count = 0
    
    # A. ArUco
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(dict_aruco, parameters)
    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            c = corners[i][0]
            cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
            if ids[i][0] == 0:
                cv2.putText(frame, "LANDING PAD", (cx-50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 2)
                print(" -> [PASS] Phat hien Bai dap ArUco ID 0")
                detected_count += 1
                
    # B. QR Code
    decoded_objects = decode(frame)
    for obj in decoded_objects:
        info = obj.data.decode('utf-8')
        pts = np.array([obj.polygon], np.int32)
        cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
        x, y = obj.rect.left, obj.rect.top
        cv2.putText(frame, f"QR: {info}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        print(f" -> [PASS] Phat hien QR Code: {info}")
        detected_count += 1
        
    # C. YOLO
    net = cv2.dnn.readNetFromDarknet("assets/models/yolov4-tiny.cfg", "assets/models/yolov4-tiny.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    with open("assets/models/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                w_box = int(detection[2] * w)
                h_box = int(detection[3] * h)
                x = int(center_x - w_box / 2)
                y = int(center_y - h_box / 2)
                boxes.append([x, y, w_box, h_box])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            label = str(classes[class_ids[i]])
            conf = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 165, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            print(f" -> [PASS] Phat hien vat the: {label} ({conf:.2f})")
            detected_count += 1
            
    cv2.imwrite("test_result.png", frame)
    print(f"[TEST] Hoan tat! Tong so vat the/muc tieu phat hien duoc: {detected_count}")
    print("[TEST] Da luu hinh anh ket qua ra file test_result.png")

if __name__ == "__main__":
    main()
