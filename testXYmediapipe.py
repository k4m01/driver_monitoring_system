import cv2
import mediapipe as mp

# เตรียม Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# เปิดใช้งานกล้อง
cap = cv2.VideoCapture(0)

# ระบุดัชนีของจุดรอบดวงตาซ้ายและขวา
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("ไม่สามารถอ่านข้อมูลจากกล้องได้")
        break

    # เปลี่ยนภาพเป็น RGB (Mediapipe ต้องการ RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ประมวลผลภาพเพื่อหาจุด Landmark
    results = face_mesh.process(rgb_frame)

    # ตรวจสอบว่าพบ Landmark หรือไม่
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # ดึงพิกัด x, y ของแต่ละจุดรอบดวงตาซ้าย
            for idx in left_eye_indices:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])

                # วาดจุดบนภาพ
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # แสดงค่าพิกัดใน Console
                print(f"Left Eye Point {idx}: (x={x}, y={y})")

            # ดึงพิกัด x, y ของแต่ละจุดรอบดวงตาขวา
            for idx in right_eye_indices:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])

                # วาดจุดบนภาพ
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

                # แสดงค่าพิกัดใน Console
                print(f"Right Eye Point {idx}: (x={x}, y={y})")

    # แสดงภาพบนหน้าจอ
    cv2.imshow('Face Mesh', frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
