import cv2
from ultralytics import YOLO

# โหลดโมเดล YOLO
model = YOLO("best_float32.tflite")

# โหลด class names
with open("coco.txt", "r") as file:
    class_list = file.read().split("\n")

# เปิดกล้อง
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# จำกัดเฟรมที่ประมวลผลต่อวินาที
frame_skip = 2  # ข้ามทุก 2 เฟรม
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ข้ามเฟรมตาม frame_skip
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # ลดขนาดเฟรมเพื่อเพิ่มความเร็ว
    small_frame = cv2.resize(frame, (320, 240))

    # ใช้โมเดล YOLO ตรวจจับวัตถุ
    results = model(small_frame)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes else []

    for detection in detections:
        x1, y1, x2, y2, conf, class_id = map(int, detection[:6])
        label = class_list[class_id] if class_id < len(class_list) else "Unknown"

        # แปลงพิกัดกลับไปยังเฟรมต้นฉบับ
        scale_x = frame.shape[1] / small_frame.shape[1]
        scale_y = frame.shape[0] / small_frame.shape[0]
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

        # วาดกรอบและข้อความบนภาพ
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # แสดงผลภาพ
    cv2.imshow("YOLO Detection", frame)

    # ออกจากลูปเมื่อกด 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
