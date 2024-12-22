import cv2
import mediapipe as mp
from scipy.spatial import distance

# Initializing Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define MAR calculation function
def calculate_mar(landmarks):
    # Points for the mouth
    upper_lip_inner = 13  # จุดบนของปาก
    lower_lip_inner = 14  # จุดล่างของปาก
    left_corner = 78      # มุมปากซ้าย
    right_corner = 308    # มุมปากขวา

    # Calculate vertical distance (ระหว่างปากบนและล่าง)
    vertical = distance.euclidean(landmarks[upper_lip_inner], landmarks[lower_lip_inner])
    
    # Calculate horizontal distance (ระหว่างมุมปากซ้าย-ขวา)
    horizontal = distance.euclidean(landmarks[left_corner], landmarks[right_corner])
    
    # MAR formula
    mar = vertical / horizontal
    return mar

# Process video input
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract the mouth landmarks
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # Calculate MAR
            mar = calculate_mar(landmarks)
            print(f"MAR: {mar:.2f}")

            # Draw landmarks for visualization
            for point in [13, 14, 61, 291]:
                cv2.circle(frame, landmarks[point], 2, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('Mouth Aspect Ratio', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
