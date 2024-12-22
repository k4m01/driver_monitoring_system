import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
import math

# กำหนดค่า Threshold และตัวแปรต่างๆ สำหรับการง่วงนอน
EAR_THRESHOLD = 0.16
MAR_THRESHOLD = 0.6
CLOSED_EYE_FRAMES = 50  # จำนวนเฟรมสำหรับการหลับตา
BLINK_FRAME_THRESHOLD = 5  # จำนวนเฟรมสำหรับการกะพริบตา 1 ครั้ง
BLINK_COUNT_THRESHOLD = 30  # จำนวนการกะพริบต่อนาทีที่จะแจ้งเตือน

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# เปิดกล้อง
video_capture = cv2.VideoCapture(0)
eye_closed_start = None
blink_count = 0
ptime = 0
start_time = time.time()

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    # print(f"Vertical Distances: A={A:.2f}, B={B:.2f}, Horizontal Distance: C={C:.2f}, EAR={ear:.2f}")
    return ear

def calculate_mar(mouth_points):
    A = distance.euclidean(mouth_points[1], mouth_points[2]) 
    B = distance.euclidean(mouth_points[0], mouth_points[3])
    mar = A / B
    # print(f"Vertical Distances: A={A:.2f}, B={B:.2f} MAR={mar:.2f}")
    return mar

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # # คำนวณ FPS ทุกๆ 1 วินาที
    # ctime = time.time()
    # fps = 1 / (ctime - ptime)
    # ptime = ctime
    # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            mouth_indices = [78, 13, 14, 308]

            left_eye = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in left_eye_indices]
            right_eye = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in right_eye_indices]
            mouth_points = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in mouth_indices]

            # print(mouth_points)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            mar = calculate_mar(mouth_points)

            # cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ตรวจจับการหลับตา
            # if ear < EAR_THRESHOLD:
            #     if eye_closed_start is None:
            #         eye_closed_start = math.floor(time.time()) % 60
            #         blink_count += 1  # นับการกระพริบตา
            #         print(f"Blink count: {blink_count}")
            #         #print(f"time start: {(math.floor(time.time()) % 60)}")
            #     elif (math.floor(time.time()) % 60) - eye_closed_start > CLOSED_EYE_FRAMES / video_capture.get(cv2.CAP_PROP_FPS):
            #         cv2.putText(frame, "DROWSINESS ALERT!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # else: >= 60:
            #     if blink_count > BLINK_COUNT_THRESHOLD:
            #         cv2.putText(frame, "FREQUENT DROWSINESS ALERT!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #         blink_count = 0
            #         start_time = time.time()

            # ตรวจจับการหาว
            #     eye_closed_start = None

            # elapsed_time = time.time() - start_time
            # print(elapsed_time)
            # if elapsed_time
            # if mar > MAR_THRESHOLD:
            #     cv2.putText(frame, "YAWNING ALERT!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
            # วาดจุดบนใบหน้า
            # y_offset = 60
            # for idx, (x, y) in enumerate(left_eye):
            #     cv2.putText(frame, f"Left Eye [{idx}]: ({x:.0f}, {y:.0f})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #     y_offset += 20

            # # วาดจุดบนใบหน้า
            # y_offset = 60
            # for idx, (x, y) in enumerate(mouth_points):
            #     cv2.putText(frame, f"Mouth [{idx}]: ({x:.0f}, {y:.0f})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #     y_offset += 20

            # y_offset = 220
            # for idx, (x, y) in enumerate(right_eye):
            #     cv2.putText(frame, f"Right Eye [{idx}]: ({x:.0f}, {y:.0f})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #     y_offset += 20
            # y_offset = 220
            # for idx, (x, y) in enumerate(right_eye):
            #     cv2.putText(frame, f"Right Eye [{idx}]: ({x:.0f}, {y:.0f})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #     y_offset += 20
            # for idx in left_eye_indices:
            #     x = int(landmarks[idx].x * frame.shape[1])
            #     y = int(landmarks[idx].y * frame.shape[0])
            #     cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            #     # print(f"Mouth Point {idx}: (x={x}, y={y})")
            # for idx in right_eye_indices:
            #     x = int(landmarks[idx].x * frame.shape[1])
            #     y = int(landmarks[idx].y * frame.shape[0])
            #     cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # วาดจุดบนใบหน้า
            for idx in mouth_indices:
                x = int(landmarks[idx].x * frame.shape[1])
                y = int(landmarks[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
    # แสดงผลและควบคุมการปิดโปรแกรม
    cv2.imshow("Drowsiness and Yawn Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()