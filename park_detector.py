from flask import Flask, Response
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = "runs/detect/train/weights/best.pt"
VIDEO_SOURCE = 0
TIME_THRESHOLD = 5
SAVE_EVIDENCE = True

ROI_POINTS = np.array([
    [200, 200],
    [600, 200],
    [600, 500],
    [200, 500]
])

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)

inside_timers = {}
alerted_ids = set()

if SAVE_EVIDENCE:
    os.makedirs("violations", exist_ok=True)


def point_in_roi(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def generate_frames():
    global inside_timers, alerted_ids

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                inside = point_in_roi((cx, cy), ROI_POINTS)

                if inside:
                    if track_id not in inside_timers:
                        inside_timers[track_id] = time.time()

                    elapsed = time.time() - inside_timers[track_id]

                    if elapsed >= TIME_THRESHOLD:
                        color = (0, 0, 255)
                        label = f"ALERT! {int(elapsed)}s"

                        if track_id not in alerted_ids:
                            print(f"🚨 Vehicle ID {track_id} violated parking rule!")
                            alerted_ids.add(track_id)

                            if SAVE_EVIDENCE:
                                filename = f"violations/violation_{track_id}_{int(time.time())}.jpg"
                                cv2.imwrite(filename, frame)
                    else:
                        color = (0, 255, 255)
                        label = f"Inside {int(elapsed)}s"

                else:
                    if track_id in inside_timers:
                        del inside_timers[track_id]
                    color = (0, 255, 0)
                    label = "Outside"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, f"ID {int(track_id)} - {label}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.polylines(frame, [ROI_POINTS], True, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Parking Violation Detection</title>
    </head>
    <body>
        <h1> Live Parking Detection</h1>
        <img src="/video" width="800">
    </body>
    </html>
    """


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)