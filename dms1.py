import os
import cv2
import numpy as np
import time
import importlib.util
import mediapipe as mp
from scipy.spatial import distance
import threading  # ใช้ threading เพื่อให้เสียงไม่ติดขัดกับการทำงานของโปรแกรม
# from playsound import playsound
import pygame
import math


# ตั้งค่าพารามิเตอร์ของโมเดล
MODEL_NAME = 'custom_model_lite'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
ALERT_FOLDER = 'alert_images'
min_conf_threshold = 0.2
resW, resH = 1280, 720
use_TPU = False

# โหลด TensorFlow Lite Runtime
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU and GRAPH_NAME == 'detect.tflite':
    GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# โหลด Label
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

# โหลดโมเดล
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# ตั้งค่าอินพุตและเอาท์พุตของโมเดล
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
boxes_idx, classes_idx, scores_idx = (1, 3, 0) if 'StatefulPartitionedCall' in output_details[0]['name'] else (0, 1, 2)

# ตั้งค่าตัวแปรสำหรับการตรวจจับง่วงนอน
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

drowsiness_alert_played = False
yawning_alert_played = False
phone_alert_played = False
cigarette_alert_played = False
bottle_alert_played = False

yawn_start_time = None
last_alert_time = None 
eye_closed_start = None
blink_count = 0
ptime = 0
yawn_count = 0

# ฟังก์ชันคำนวณ EAR
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ฟังก์ชันคำนวณ MAR
def calculate_mar(mouth_points):
    A = distance.euclidean(mouth_points[1], mouth_points[2]) 
    B = distance.euclidean(mouth_points[0], mouth_points[3])
    mar = A / B
    # print(f"Vertical Distances: A={A:.2f}, B={B:.2f} MAR={mar:.2f}")
    return mar

pygame.mixer.init()

def play_alert_sound(file_name):
    sound = pygame.mixer.Sound(file_name)
    sound.play()
# def play_alert_sound():
#     sound = pygame.mixer.Sound('alarm.wav')
#     sound.play()


    

def capture_frame(frame, alert_type):
    # Create folder if it doesn't exist
    if not os.path.exists(ALERT_FOLDER):
        os.makedirs(ALERT_FOLDER)
        
    # Create a filename with a timestamp and alert type
    timestamp = time.strftime("%Y_%m_%d_%H %M %S")
    filename = f"{alert_type}_{timestamp}.jpg"
    file_path = os.path.join(ALERT_FOLDER, filename)
    
    # Save the image and print the file path
    cv2.imwrite(file_path, frame)
    print(f"Captured frame for {alert_type} alert as {file_path}")


# เริ่มต้นการจับภาพจากกล้อง
video_capture = cv2.VideoCapture(0)
video_capture.set(3, resW)
video_capture.set(4, resH)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(rgb_frame, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    phone_position = None
    cigarette_position = None
    bottle_position = None
    
    for i in range(len(scores)):
        if scores[i] > min_conf_threshold:
            ymin = int(max(1, boxes[i][0] * resH))
            xmin = int(max(1, boxes[i][1] * resW))
            ymax = int(min(resH, boxes[i][2] * resH))
            xmax = int(min(resW, boxes[i][3] * resW))

            object_label = labels[int(classes[i])]
            label = f'{object_label}: {int(scores[i] * 100)}%'
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            if object_label == 'phone_cell':
                phone_position = ((xmin + xmax) // 2, (ymin + ymax) // 2)
                #print(phone_position)
            if object_label == 'cigarette':
                cigarette_position = ((xmin + xmax) // 2, (ymin + ymax) // 2)
            if object_label == 'bottle':
                bottle_position = ((xmin + xmax) // 2, (ymin + ymax) // 2)

    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            mouth_indices = [78, 13, 14, 308]
            left_ear_landmark = 234  
            right_ear_landmark = 454

            left_eye = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in left_eye_indices]
            right_eye = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in right_eye_indices]
            mouth_points = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in mouth_indices]
            left_ear_coords = (int(landmarks[left_ear_landmark].x * frame.shape[1]), int(landmarks[left_ear_landmark].y * frame.shape[0]))
            right_ear_coords = (int(landmarks[right_ear_landmark].x * frame.shape[1]), int(landmarks[right_ear_landmark].y * frame.shape[0]))

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(mouth_points)

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                    # print(video_capture.get(cv2.CAP_PROP_FPS))
                elif (time.time() - eye_closed_start) > CLOSED_EYE_FRAMES / video_capture.get(cv2.CAP_PROP_FPS):
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not drowsiness_alert_played:  # ตรวจสอบว่าการแจ้งเตือนนี้ยังไม่เคยเล่น
                        threading.Thread(target=play_alert_sound, args=('stop.wav',)).start()
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
                    elapsed_time = time.time() - yawn_start_time #การคำนวณระยะเวลาที่ผ่านไปตั้งแต่เริ่มนับหาวครั้งแรก
                    
                    if elapsed_time <= YAWN_ALERT_INTERVAL:
                        yawn_count += 1
                        print(f"ํYawn {yawn_count}")
                        if yawn_count >= MAX_YAWN_COUNT:
                            if last_alert_time is None or time.time() - last_alert_time > 0.5:
                                threading.Thread(target=play_alert_sound, args=('stop.wav',)).start()   # เล่นเสียงแจ้งเตือน
                                capture_frame(frame, "Yawning")  # บันทึกภาพ
                                last_alert_time = time.time()
                                # print("แจ้งเตือน: หาว 3 ครั้งใน 1 นาที!")
                    else:
                        yawn_count = 0
                        yawn_start_time = None
                        last_alert_time = None
            else:
                yawning_alert_played = False

            # if mar > MAR_THRESHOLD:
            #     cv2.putText(frame, "YAWNING ALERT!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     if not yawning_alert_played:
            #         threading.Thread(target=play_alert_sound).start()  # เล่นเสียงโดยใช้ threading
            #         capture_frame(frame, "Yawning")
            #         yawning_alert_played = True
            # else:
            #     yawning_alert_played = False

            if phone_position:
                distance_left = distance.euclidean(phone_position, left_ear_coords)
                distance_right = distance.euclidean(phone_position, right_ear_coords)
                # print(f"{distance_right} < {PHONE_NEAR_EAR_THRESHOLD} ")
                if distance_left < PHONE_NEAR_EAR_THRESHOLD * frame.shape[1] or distance_right < PHONE_NEAR_EAR_THRESHOLD * frame.shape[1]:
                    cv2.putText(frame, "PHONE ALERT!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if not phone_alert_played:
                        print(threading.Thread(target=play_alert_sound, args=('phone_detected.wav',)).start())
                        capture_frame(frame, "Phonecall")
                        phone_alert_played = True
                else:
                    phone_alert_played = False


            if cigarette_position:
                distance_mouth = distance.euclidean(cigarette_position,mouth_points[1])
                if distance_mouth < CIGARETTE_NEAR_MOUTH_THRESHOLD * frame.shape[1]:
                    cv2.putText(frame, "CIGARETTE ALERT!", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if not cigarette_alert_played:
                        threading.Thread(target=play_alert_sound, args=('cigarette_detected.wav',)).start()
                        capture_frame(frame, "Cigarette")
                        cigarette_alert_played = True
                else:
                    cigarette_alert_played = False

            if bottle_position:
                distance_mouth1 = distance.euclidean(bottle_position,mouth_points[1])
                # print(distance_mouth1)
                if distance_mouth1 < BOTTLE_NEAR_MOUTH_THRESHOLD * frame.shape[1]:
                    cv2.putText(frame, "DRINKING WATER ALERT!", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if not bottle_alert_played:
                        threading.Thread(target=play_alert_sound, args=('bottle_detected.wav',)).start()
                        capture_frame(frame, "DrinkWater")
                        bottle_alert_played = True
                else:
                    bottle_alert_played = False

            # # วาดจุดบนใบหน้า
            # for idx in left_eye_indices + right_eye_indices + mouth_indices:
            #     x = int(landmarks[idx].x * frame.shape[1])
            #     y = int(landmarks[idx].y * frame.shape[0])
            #     cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
           

    ctime = time.time()

    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
