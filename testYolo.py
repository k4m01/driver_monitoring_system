from ultralytics import YOLO
import cv2
import winsound  # ใช้สำหรับเล่นเสียงแจ้งเตือน (Windows)

# โหลดโมเดล YOLOv8
model = YOLO('best.pt')  # ใช้โมเดล YOLOv8n หรือโมเดลที่เทรนเอง

# เปิดกล้อง (กล้องตัวที่ 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

while True:
    # อ่านเฟรมจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        break

    # ใช้โมเดล YOLOv8 ทำนายผล
    results = model(frame)

    # ดึงข้อมูลผลลัพธ์การตรวจจับ
    detections = results[0].boxes.data  # ใช้ข้อมูล bounding box
    for detection in detections:
        # ดึง class ID และความมั่นใจ
        class_id = int(detection[5])  # สมมติ class ID อยู่ใน index 5
        confidence = detection[4]    # สมมติ confidence อยู่ใน index 4
        
        # ตรวจสอบว่า class ID เป็นการสูบบุหรี่ (แทนที่ด้วย class ID ที่เหมาะสม)
        if class_id == 1 and confidence > 0.5:  # เปลี่ยน '1' เป็น class ID ของการสูบบุหรี่
            print("ตรวจพบการสูบบุหรี่!")
            # เล่นเสียงแจ้งเตือน
            winsound.Beep(1000, 500)  # เล่นเสียงที่ความถี่ 1000 Hz นาน 500 ms

            # แสดงข้อความบนภาพ
            cv2.putText(frame, "Smoking Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)

    # วาดผลลัพธ์บนภาพ
    annotated_frame = results[0].plot()

    # แสดงผลภาพ
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # กด 'q' เพื่อหยุด
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการทำงานของกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
