import cv2
import mediapipe as mp
from scipy.spatial import distance

# เตรียม Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# เปิดใช้งานกล้อง
cap = cv2.VideoCapture(0)

# ระบุดัชนีของจุดรอบดวงตาซ้ายและขวา
lip_indices = [78 ,81 ,13 ,311 ,308 ,402 ,14 ,178] 

#78 ด้านซ้าย P1
#308 ด้านขวา P5
#13 บน P3
#14 ล่าง P7
#178 ล่างซ่้าย P8
#402 ล่างขวา P6
#81 บนซ้าย P2
#311 บนล่าง P4


# ฟังก์ชันคำนวณ EAR
def calculate_mar(mouth_points):
    A = distance.euclidean(mouth_points[3], mouth_points[5])  # p2 - p8
    B = distance.euclidean(mouth_points[2], mouth_points[6])  # p3 - p7
    C = distance.euclidean(mouth_points[1], mouth_points[7])  # p4 - p6
    D = distance.euclidean(mouth_points[0], mouth_points[4])  # p1 - p5
    mar = (A + B + C) / (2.0 * D)
    # print(f"Vertical Distances: A={A:.2f}, B={B:.2f} MAR={mar:.2f}")
    return mar

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
            landmarks = face_landmarks.landmark
            # ดึงพิกัดจุด Landmark ของดวงตา
            mouth_points = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in lip_indices]
            height, width, _ = frame.shape

            mar = calculate_mar(mouth_points)

            # แสดงค่าของ EAR
            cv2.putText(frame, f"Left EAR: {mar:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # qๆ
            # แสดงพิกัดโดยขึ้นบรรทัดใหม่
            # y_offset = 60
            # for idx, (x, y) in enumerate(left_eye_points):
            #     cv2.putText(frame, f"Left Eye [{idx}]: ({x}, {y})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #     y_offset += 20

            # y_offset = 220
            # for idx, (x, y) in enumerate(right_eye_points):
            #     cv2.putText(frame, f"Right Eye [{idx}]: ({x}, {y})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #     y_offset += 20

            # วาดจุด Landmark รอบดวงตา
            for idx in lip_indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                # วาดจุด Landmark
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # แสดงภาพบนหน้าจอ
    cv2.imshow('Face Mesh', frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
