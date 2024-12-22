import os
import cv2
import numpy as np
import time
import mediapipe as mp
from scipy.spatial import distance
import threading
import pygame

# ตั้งค่าค่าคงที่
MAR_THRESHOLD = 0.6  # ค่าที่บ่งชี้การอ้าปาก (หาว)
MAX_YAWN_COUNT = 5   # จำนวนการหาวที่ถือว่าเกิน
ALERT_FOLDER = 'alert_images'
YAWN_ALERT_INTERVAL = 10  # ระยะเวลาในการตรวจสอบ (1 นาที)

# ฟังก์ชันคำนวณ MAR
def calculate_mar(mouth_points):
    A = distance.euclidean(mouth_points[1], mouth_points[2]) 
    B = distance.euclidean(mouth_points[0], mouth_points[3])
    return A / B

# ฟังก์ชันเล่นเสียงแจ้งเตือน
pygame.mixer.init()

def play_alert_sound():
    sound = pygame.mixer.Sound('alarm.wav')  # ไฟล์เสียงแจ้งเตือน
    sound.play()

# ฟังก์ชันบันทึกภาพแจ้งเตือน
def capture_frame(frame, alert_type):
    if not os.path.exists(ALERT_FOLDER):
        os.makedirs(ALERT_FOLDER)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"{alert_type}_{timestamp}.jpg"
    file_path = os.path.join(ALERT_FOLDER, filename)
    cv2.imwrite(file_path, frame)
    print(f"Captured frame for {alert_type} alert: {file_path}")

# เริ่มต้น Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# เริ่มต้นการจับภาพ
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 1280)
video_capture.set(4, 720)

yawn_count = 0
yawn_start_time = None
mouth_open = False  # สถานะปากปิดเริ่มต้น
alert_active = False  # สถานะแจ้งเตือนเริ่มต้นเป็น False
last_alert_time = None  # เวลาของการแจ้งเตือนครั้งล่าสุด

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            mouth_indices = [78, 13, 14, 308]
            mouth_points = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in mouth_indices]
            mar = calculate_mar(mouth_points)

            # แสดงค่า MAR บนหน้าจอ
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ตรวจจับการเริ่มอ้าปาก (เปลี่ยนสถานะจากปิดเป็นเปิด)
            if mar > MAR_THRESHOLD:
                if not mouth_open:  # หากก่อนหน้านี้ปากปิด
                    mouth_open = True  # เปลี่ยนสถานะเป็นปากเปิด
                    if yawn_start_time is None:
                        yawn_start_time = time.time()  # เริ่มจับเวลา
                    elapsed_time = time.time() - yawn_start_time

                    if elapsed_time <= YAWN_ALERT_INTERVAL:  # หากยังอยู่ในช่วงเวลา 1 นาที
                        yawn_count += 1
                        print(f"Yawn count: {yawn_count} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")  # ปริ้นเวลา
                        
                        # หากหาวครบ 5 ครั้งแล้ว
                        if yawn_count >= MAX_YAWN_COUNT:
                            # เช็คว่าเวลาผ่านไปตั้งแต่การแจ้งเตือนครั้งล่าสุดมากกว่า 0.5 วินาทีหรือยัง
                            if last_alert_time is None or time.time() - last_alert_time > 0.5:
                                print(f"แจ้งเตือน: หาวครบ 5 ครั้ง! at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")  # ปริ้นเวลา
                                threading.Thread(target=play_alert_sound).start()
                                capture_frame(frame, "Yawning")
                                last_alert_time = time.time()  # อัปเดตเวลาของการแจ้งเตือนครั้งล่าสุด
                    else:
                        # รีเซ็ตค่าทั้งหมดเมื่อครบ 1 นาที
                        yawn_count = 0
                        yawn_start_time = None
                        last_alert_time = None
        
            else:
                mouth_open = False  # หาก MAR ต่ำกว่าค่าที่กำหนด เปลี่ยนสถานะเป็นปากปิด

    # แสดงภาพ
    cv2.imshow('Yawning Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
