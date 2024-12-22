import cv2
import mediapipe as mp
from scipy.spatial import distance

# เตรียม Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# เปิดใช้งานกล้อง
cap = cv2.VideoCapture(0)

# ระบุดัชนีของจุดรอบดวงตาซ้ายและขวา
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

# ฟังก์ชันคำนวณ EAR
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ฟังก์ชันดึงพิกัดจุด Landmark
def get_eye_points(eye_indices, landmarks, frame_shape):
    return [(int(landmarks[idx].x * frame_shape[1]), int(landmarks[idx].y * frame_shape[0])) for idx in eye_indices]

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
            # ดึงพิกัดจุด Landmark ของดวงตา
            left_eye_points = get_eye_points(left_eye_indices, face_landmarks.landmark, frame.shape)
            right_eye_points = get_eye_points(right_eye_indices, face_landmarks.landmark, frame.shape)

            # คำนวณ EAR สำหรับดวงตาซ้ายและขวา
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)

            ear = (left_ear + right_ear) / 2.0

            # แสดงค่าของ EAR
            # cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # qๆ
            # แสดงพิกัดโดยขึ้นบรรทัดใหม่
            y_offset = 60
            for idx, (x, y) in enumerate(left_eye_points):
                cv2.putText(frame, f"Left Eye [{idx}]: ({x}, {y})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                y_offset += 20

            y_offset = 220
            for idx, (x, y) in enumerate(right_eye_points):
                cv2.putText(frame, f"Right Eye [{idx}]: ({x}, {y})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20

            # วาดจุด Landmark รอบดวงตา
            for point in left_eye_points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)
            for point in right_eye_points:
                cv2.circle(frame, point, 2, (0, 0, 255), -1)

    # แสดงภาพบนหน้าจอ
    cv2.imshow('Face Mesh', frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
