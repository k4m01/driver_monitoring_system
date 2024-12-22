import cv2
import time
import os
import argparse
import numpy as np
from threading import Thread
import importlib.util

from facial_tracking.facialTracking import FacialTracker
import facial_tracking.conf as conf

class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
        self.stream.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', required=True)
    parser.add_argument('--graph', default='detect.tflite')
    parser.add_argument('--labels', default='labelmap.txt')
    parser.add_argument('--threshold', default=0.4)
    parser.add_argument('--resolution', default='1280x720')
    parser.add_argument('--edgetpu', action='store_true')
    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = map(int, args.resolution.split('x'))
    use_TPU = args.edgetpu

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

    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if labels[0] == '???':
        del(labels[0])

    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = (input_details[0]['dtype'] == np.float32)

    boxes_idx, classes_idx, scores_idx = (1, 3, 0) if 'StatefulPartitionedCall' in output_details[0]['name'] else (0, 1, 2)

    videostream = VideoStream(resolution=(resW, resH), framerate=30).start()
    time.sleep(1)
    facial_tracker = FacialTracker()
    ptime = 0

    while True:
        frame1 = videostream.read()
        frame = frame1.copy()

        # Facial tracking
        facial_tracker.process_frame(frame)

        # Display FPS and facial tracker results
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {int(fps)}', (30,30), 0, 0.6, conf.TEXT_COLOR, 1, lineType=cv2.LINE_AA)
        
        # Object detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        for i in range(len(scores)):
            if min_conf_threshold < scores[i] <= 1.0:
                ymin = int(max(1, boxes[i][0] * resH))
                xmin = int(max(1, boxes[i][1] * resW))
                ymax = int(min(resH, boxes[i][2] * resH))
                xmax = int(min(resW, boxes[i][3] * resW))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                label = f'{labels[int(classes[i])]}: {int(scores[i] * 100)}%'
                cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if facial_tracker.detected:
            cv2.putText(frame, f'{facial_tracker.eyes_status}', (30,70), 0, 0.8, conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f'{facial_tracker.yawn_status}', (30,110), 0, 0.8, conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Facial and Object Tracking', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    videostream.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
