import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
import threading
import pygame
import os
import math

# กำหนดค่า Threshold และตัวแปรต่างๆ สำหรับการง่วงนอน
EAR_THRESHOLD = 0.16
MAR_THRESHOLD = 0.6
CLOSED_EYE_FRAMES = 60
BLINK_FRAME_THRESHOLD = 2  # จำนวนเฟรมสำหรับการกะพริบตา 1 ครั้ง
BLINK_COUNT_THRESHOLD = 30 # จำนวนการกะพริบต่อนาทีที่จะแจ้งเตือน
BLINK_TIMER_THRESHOLD  = 61 # จับเวลาใน 1 นาที
YAWN_COUNT_THRESHOLD = 5 # จำนวนการกะพริบต่อนาทีที่จะแจ้งเตือน
YAWN_FRAME_THRESHOLD = 50 # จำนวนเฟรมสำหรับการหาว 1 ครั้ง
YAWN_TIMER_THRESHOLD = 60 # จับเวลาใน 1 นาที
ALERT_FOLDER = 'alert_images'

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# เปิดกล้อง
video_capture = cv2.VideoCapture(0)
eye_closed_start = None
yawn_count = 0
yawn_frames = 0
yawn_start_time = int(time.monotonic())
blink_count = 0
blink_frames = 0
ptime = 0
eye_start_time = int(time.monotonic())  
last_alert_time = int(time.monotonic())
yawning_alert_played = False

pygame.mixer.init()

def play_alert_sound(file_name):
    sound = pygame.mixer.Sound(file_name)
    sound.play()
    print(f"Playing sound: {file_name}")

def capture_frame(frame, alert_type):
    if not os.path.exists(ALERT_FOLDER):
        os.makedirs(ALERT_FOLDER)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"{alert_type}_{timestamp}.jpg"
    file_path = os.path.join(ALERT_FOLDER, filename)
    cv2.imwrite(file_path, frame)
    print(f"Captured frame for {alert_type} alert as {file_path}")

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
            mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178]

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
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Bilnk Count: {blink_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Bilnk frames: {blink_frames}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yawn Count {yawn_count}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yawn frames {yawn_frames}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ตรวจจับการหลับตา
     
            # if ear < EAR_THRESHOLD:   
            #     blink_frames += 1
            # else:
            #     if blink_frames >= BLINK_FRAME_THRESHOLD:
            #         blink_count += 1  
            #     blink_frames = 0

            # if elapsed_time >= 60:
            #     start_time = time.monotonic()  # รีเซ็ตตัวจับเวลา
            #     if blink_count >= BLINK_COUNT_THRESHOLD and (time.monotonic() - last_alert_time) >= 60:
            #         cv2.putText(frame, "ALERT", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #         last_alert_time = time.monotonic()  # ป้องกันการแจ้งเตือนซ้ำ
            #     blink_count = 0

            elapsed_time_eyes = int(time.monotonic() - eye_start_time)
            cv2.putText(frame, f"Timer Eyes: {elapsed_time_eyes}", (210, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if ear < EAR_THRESHOLD:
                blink_frames += 1
                if eye_closed_start is None:
                    eye_closed_start = int(time.monotonic())
                elif (int(time.monotonic()) - eye_closed_start) > CLOSED_EYE_FRAMES / video_capture.get(cv2.CAP_PROP_FPS):
                    cv2.putText(frame, "DROWSINESS ALERT!", (210, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not drowsiness_alert_played:
                        threading.Thread(target=play_alert_sound, args=('./AlertSound/stop.wav',)).start()
                        capture_frame(frame, "Drowsiness")
                        drowsiness_alert_played = True
            else:
                if blink_frames >= BLINK_FRAME_THRESHOLD:
                    blink_count += 1
                if elapsed_time_eyes >= BLINK_TIMER_THRESHOLD:  
                    if blink_count >= BLINK_COUNT_THRESHOLD and int(time.monotonic()) - last_alert_time >= BLINK_TIMER_THRESHOLD:
                        cv2.putText(frame, "DROWSINESS ALERT!", (210, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print("OK")
                        last_alert_time = int(time.monotonic()) 
                        if not drowsiness_alert_played:
                            threading.Thread(target=play_alert_sound, args=('./AlertSound/stop.wav',)).start()
                            capture_frame(frame, "Drowsiness")
                            drowsiness_alert_played = True   
                    eye_start_time = int(time.monotonic())  
                    blink_count = 0
                blink_frames = 0
                eye_closed_start = None
                drowsiness_alert_played = False
                
            # if mar > MAR_THRESHOLD:
            #     yawn_frames += 1
            # else:
            #     if yawn_frames >= YAWN_FRAME_THRESHOLD:
            #         yawn_count += 1
            #     yawn_frames = 0

            # elapsed_time = int(time.monotonic() - start_time)
            # cv2.putText(frame, f"Timer : {elapsed_time}", (210, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # if elapsed_time <= 60:
            #     start_time = int(time.monotonic())  # รีเซ็ตตัวจับเวลา
            #     if yawn_count > YAWN_COUNT_THRESHOLD and (int(time.monotonic()) - last_alert_time) >= 60:
            #         cv2.putText(frame, "ALERT", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #         print("OK")
            #         last_alert_time = int(time.monotonic())  # ป้องกันการแจ้งเตือนซ้ำ
            #     yawn_count = 0

            elapsed_time_yawn = int(time.monotonic() - yawn_start_time)  
            cv2.putText(frame, f"Timer Yawn: {elapsed_time_yawn}", (210, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if mar > MAR_THRESHOLD:
                yawn_frames += 1  
            
            else:
                if elapsed_time_yawn <= YAWN_TIMER_THRESHOLD:
                    if yawn_frames >= YAWN_FRAME_THRESHOLD:  
                        yawn_count += 1  
                        if yawn_count >= YAWN_COUNT_THRESHOLD:  
                            if last_alert_time is None or time.monotonic() - last_alert_time > 0.5:
                                threading.Thread(target=play_alert_sound, args=('./AlertSound/stop.wav',)).start()
                                capture_frame(frame, "Yawning")
                                last_alert_time = time.monotonic() 
                else:
                    yawn_start_time = int(time.monotonic()) 
                    yawn_count = 0
                yawning_alert_played = False 
                yawn_frames = 0
                
                  

            # minute_timer_start  =  int(time.monotonic())
            # if int(time.monotonic()) - minute_timer_start >= 60:
            #     yawn_count = 0  
            #     yawn_frames = 0  
            #     minute_timer_start = int(time.monotonic())  

                    

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