from ultralytics import YOLO
import cv2

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

    # วาดผลลัพธ์บนภาพ
    annotated_frame = results[0].plot()  # เพิ่ม bounding box และ label ลงบนภาพ

    # แสดงผลภาพ
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # กด 'q' เพื่อหยุด
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการทำงานของกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
