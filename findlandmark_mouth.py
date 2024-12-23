import cv2
import mediapipe as mp

# ตั้งค่า Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# เริ่มต้นการจับภาพจากกล้อง
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # เปลี่ยนสีภาพเป็น RGB ตามที่ Mediapipe ต้องการ
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ประมวลผลภาพและตรวจจับ Landmark
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark บนริมฝีปาก: indices [61, 146, 91, 181, 84, 17, 314, 405]
            lip_indices = [78, 178, 81, 13, 14, 311 ,402, 308]
            height, width, _ = frame.shape
            
            for idx in lip_indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                # วาดจุด Landmark
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    # แสดงผลลัพธ์
    cv2.imshow("Lip Landmarks", frame)
    
    # กด 'q' เพื่อออกจากการทำงาน
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
