import os
import cv2
import numpy as np
import time
import mediapipe as mp
from scipy.spatial import distance
import threading
import pygame
from ultralytics import YOLO

# ตั้งค่าพารามิเตอร์
ALERT_FOLDER = 'alert_images'
EAR_THRESHOLD = 0.16
MAR_THRESHOLD = 0.6
CLOSED_EYE_FRAMES = 60
PHONE_NEAR_EAR_THRESHOLD = 0.1
CIGARETTE_NEAR_MOUTH_THRESHOLD = 0.1
BOTTLE_NEAR_MOUTH_THRESHOLD = 0.1
MAX_YAWN_COUNT = 5
YAWN_ALERT_INTERVAL = 60

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

pygame.mixer.init()

def play_alert_sound(file_name):
    sound = pygame.mixer.Sound(file_name)
    sound.play()

def capture_frame(frame, alert_type):
    if not os.path.exists(ALERT_FOLDER):
        os.makedirs(ALERT_FOLDER)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"{alert_type}_{timestamp}.jpg"
    file_path = os.path.join(ALERT_FOLDER, filename)
    cv2.imwrite(file_path, frame)
    print(f"Captured frame for {alert_type} alert as {file_path}")

# ฟังก์ชันคำนวณ EAR
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ฟังก์ชันคำนวณ MAR
def calculate_mar(mouth_points):
    A = distance.euclidean(mouth_points[3], mouth_points[5])
    B = distance.euclidean(mouth_points[2], mouth_points[6])
    C = distance.euclidean(mouth_points[1], mouth_points[7])
    D = distance.euclidean(mouth_points[0], mouth_points[4])
    return (A + B + C) / (2.0 * D)

# โหลดโมเดล YOLO
model = YOLO('best.pt')

# เริ่มต้นการจับภาพจากกล้อง
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

ptime = 0
eye_closed_start = None
drowsiness_alert_played = False
yawning_alert_played = False
phone_alert_played = False
cigarette_alert_played = False
bottle_alert_played = False
yawn_count = 0
yawn_start_time = None
last_alert_time = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)

    # วาด bounding box ของ YOLO
    for detection in results[0].boxes:
        cls = int(detection.cls[0])
        conf = detection.conf[0]
        x1, y1, x2, y2 = map(int, detection.xyxy[0])

        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if model.names[cls] == 'phone_cell' and conf > PHONE_NEAR_EAR_THRESHOLD:
            cv2.putText(frame, "PHONE DETECTED!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not phone_alert_played:
                threading.Thread(target=play_alert_sound, args=('AlertSound/phone_alert.wav',)).start()
                capture_frame(frame, "Phone")
                phone_alert_played = True

        elif model.names[cls] == 'cigarette' and conf > CIGARETTE_NEAR_MOUTH_THRESHOLD:
            cv2.putText(frame, "CIGARETTE DETECTED!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not cigarette_alert_played:
                threading.Thread(target=play_alert_sound, args=('AlertSound/cigarette_alert.wav',)).start()
                capture_frame(frame, "Cigarette")
                cigarette_alert_played = True

        elif model.names[cls] == 'bottle' and conf > BOTTLE_NEAR_MOUTH_THRESHOLD:
            cv2.putText(frame, "BOTTLE DETECTED!", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not bottle_alert_played:
                threading.Thread(target=play_alert_sound, args=('AlertSound/bottle_detected.wav',)).start()
                capture_frame(frame, "Bottle")
                bottle_alert_played = True

    # ตรวจจับใบหน้า
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178]

            left_eye = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in left_eye_indices]
            right_eye = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in right_eye_indices]
            mouth_points = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in mouth_indices]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = calculate_mar(mouth_points)

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif (time.time() - eye_closed_start) > CLOSED_EYE_FRAMES / video_capture.get(cv2.CAP_PROP_FPS):
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not drowsiness_alert_played:
                        threading.Thread(target=play_alert_sound, args=('AlertSound/drowsiness_alert.wav',)).start()
                        capture_frame(frame, "Drowsiness")
                        drowsiness_alert_played = True
            else:
                eye_closed_start = None
                drowsiness_alert_played = False

            if mar > MAR_THRESHOLD:
                if not yawning_alert_played:
                    yawning_alert_played = True
                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    elapsed_time = time.time() - yawn_start_time
                    if elapsed_time <= YAWN_ALERT_INTERVAL:
                        yawn_count += 1
                        if yawn_count >= MAX_YAWN_COUNT:
                            if last_alert_time is None or time.time() - last_alert_time > 0.5:
                                threading.Thread(target=play_alert_sound, args=('AlertSound/yawning_alert.wav',)).start()
                                capture_frame(frame, "Yawning")
                                last_alert_time = time.time()
                    else:
                        yawn_count = 0
                        yawn_start_time = None
            else:
                yawning_alert_played = False

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Drowsiness Detection with YOLO', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()