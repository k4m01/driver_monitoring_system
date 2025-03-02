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
CLOSED_EYE_FRAMES = 30
BLINK_FRAME_THRESHOLD = 1  # จำนวนเฟรมสำหรับการกะพริบตา 1 ครั้ง
BLINK_COUNT_THRESHOLD = 30 # จำนวนการกะพริบต่อนาทีที่จะแจ้งเตือน
BLINK_TIMER_THRESHOLD  = 61 # จับเวลาใน 1 นาที
YAWN_COUNT_THRESHOLD = 5 # จำนวนการกะพริบต่อนาทีที่จะแจ้งเตือน
YAWN_FRAME_THRESHOLD = 11 # จำนวนเฟรมสำหรับการหาว 1 ครั้ง
YAWN_TIMER_THRESHOLD = 60 # จับเวลาใน 1 นาที0
PHONE_NEAR_EAR_THRESHOLD = 0.1
CIGARETTE_NEAR_MOUTH_THRESHOLD = 0.1
BOTTLE_NEAR_MOUTH_THRESHOLD = 0.1
MAX_YAWN_COUNT = 5 
YAWN_TIMER_THRESHOLD = 60

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
model = YOLO('best(4).pt')

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
yawn_frames = 0
blink_frames = 0
blink_count = 0
yawn_start_time = int(time.monotonic())
last_alert_time = int(time.monotonic())
eye_start_time = int(time.monotonic()) 
elapsed_time = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)

    phone_position = None
    cigarette_position = None
    bottle_position = None
    canned_position = None

    # วาด bounding box ของ YOLO
    for detection in results[0].boxes:
        cls = int(detection.cls[0])
        conf = detection.conf[0]
        x1, y1, x2, y2 = map(int, detection.xyxy[0])

        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if model.names[cls] == 'phone':
            phone_position = ((x1 + x2) // 2, (y1 + y2) // 2)

        elif model.names[cls] == 'cigarette':
            cigarette_position = ((x1 + x2) // 2, (y1 + y2) // 2)

        elif model.names[cls] == 'bottle':
            bottle_position =  ((x1 + x2) // 2, (y1 + y2) // 2)

        elif model.names[cls] == 'canned':
            canned_position =  ((x1 + x2) // 2, (y1 + y2) // 2)
        
    # ตรวจจับใบหน้า
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178]    #78 ด้านซ้าย P1
                                                                    #308 ด้านขวา P5
                                                                    #13 บน P3
                                                                    #14 ล่าง P7
                                                                    #178 ล่างซ่้าย P8
                                                                    #402 ล่างขวา P6
                                                                    #81 บนซ้าย P2
                                                                    #311 บนล่าง P4

            left_ear_landmark = 234  
            right_ear_landmark = 454
            mouth_landmark = 14

            left_eye = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in left_eye_indices]
            right_eye = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in right_eye_indices]
            mouth_points = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in mouth_indices]
            left_ear_coords = (int(landmarks[left_ear_landmark].x * frame.shape[1]), int(landmarks[left_ear_landmark].y * frame.shape[0]))
            right_ear_coords = (int(landmarks[right_ear_landmark].x * frame.shape[1]), int(landmarks[right_ear_landmark].y * frame.shape[0]))
            mouth_ear_coords = (int(landmarks[mouth_landmark].x * frame.shape[1]), int(landmarks[mouth_landmark].y * frame.shape[0]))

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = calculate_mar(mouth_points)

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Bilnk Count: {blink_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Bilnk frames: {blink_frames}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yawn Count {yawn_count}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yawn frames {yawn_frames}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
                        # print("OK")
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

            #ส่วนหาว
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

            if phone_position:
                distance_left = distance.euclidean(phone_position, left_ear_coords)
                distance_right = distance.euclidean(phone_position, right_ear_coords)
                # print(f"{distance_right} < {PHONE_NEAR_EAR_THRESHOLD} ")
                if distance_left < PHONE_NEAR_EAR_THRESHOLD * frame.shape[1] or distance_right < PHONE_NEAR_EAR_THRESHOLD * frame.shape[1]:
                    cv2.putText(frame, "PHONE ALERT!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if not phone_alert_played:
                        print(threading.Thread(target=play_alert_sound, args=('./AlertSound/phone_detected.wav',)).start())
                        capture_frame(frame, "Phonecall")
                        phone_alert_played = True
                else:
                    phone_alert_played = False


            if cigarette_position:
                for mouth_point in mouth_points:
                    distance_mouth = distance.euclidean(cigarette_position, mouth_point)
                    if distance_mouth < CIGARETTE_NEAR_MOUTH_THRESHOLD * frame.shape[1]:
                        cv2.putText(frame, "CIGARETTE ALERT!", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        if not cigarette_alert_played:
                            threading.Thread(target=play_alert_sound, args=('./AlertSound/cigarette_detected.wav',)).start()
                            capture_frame(frame, "Cigarette")
                            cigarette_alert_played = True
                        break
                    else:
                        cigarette_alert_played = False

            if bottle_position or canned_position:
                for mouth_point in mouth_points:
                    if bottle_position and distance.euclidean(bottle_position, mouth_point) < BOTTLE_NEAR_MOUTH_THRESHOLD * frame.shape[1]:
                        cv2.putText(frame, "DRINKING WATER ALERT!", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        if not bottle_alert_played:
                            threading.Thread(target=play_alert_sound, args=('./AlertSound/bottle_detected.wav',)).start()
                            capture_frame(frame, "DrinkWater")
                            bottle_alert_played = True
                        break 

                    if canned_position and distance.euclidean(canned_position, mouth_point) < BOTTLE_NEAR_MOUTH_THRESHOLD * frame.shape[1]:
                        cv2.putText(frame, "DRINKING WATER ALERT!", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        if not bottle_alert_played:
                            threading.Thread(target=play_alert_sound, args=('./AlertSound/bottle_detected.wav',)).start()
                            capture_frame(frame, "DrinkWater")
                            bottle_alert_played = True
                        break  
                else:
                    bottle_alert_played = False 

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Drowsiness Detection with YOLO', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
