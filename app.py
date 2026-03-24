from flask import Flask, Response, request, redirect, url_for, render_template_string
import cv2
import numpy as np
import time
import os
import threading
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Configuration via environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "runs/detect/train/weights/best.pt")
TIME_THRESHOLD = int(os.environ.get("TIME_THRESHOLD", "5"))
SAVE_EVIDENCE = os.environ.get("SAVE_EVIDENCE", "True").lower() == "true"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if SAVE_EVIDENCE:
    os.makedirs("violations", exist_ok=True)

print(f"📦 Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("✅ Model loaded!")

ROI_POINTS = np.array([
    [200, 200],
    [600, 200],
    [600, 500],
    [200, 500]
])

# Global state
current_video_path = None
video_lock = threading.Lock()
inside_timers = {}
alerted_ids = set()

def point_in_roi(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def generate_frames():
    global inside_timers, alerted_ids

    while True:
        with video_lock:
            video_path = current_video_path

        if video_path is None or not os.path.exists(video_path):
            # Return a placeholder frame if no video is available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No video uploaded yet.", (80, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Go to / to upload a video.", (60, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            time.sleep(2)
            continue

        print(f"▶️  Playing video: {video_path}")
        frame_count = 0
        inside_timers = {}
        alerted_ids = set()

        while True:
            success, frame = cap.read()
            if not success:
                # Loop the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                inside_timers = {}
                alerted_ids = set()
                continue

            # Check if video changed
            with video_lock:
                if current_video_path != video_path:
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
                                    filename = f"violations/violation_{int(track_id)}_{int(time.time())}.jpg"
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
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.polylines(frame, [ROI_POINTS], True, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            frame_count += 1

        cap.release()


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Illegal Parking Detection</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            min-height: 100vh;
        }
        header {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            padding: 20px 40px;
            border-bottom: 2px solid #e94560;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        header h1 { font-size: 1.6rem; color: #fff; }
        header span { font-size: 2rem; }
        .container { max-width: 1200px; margin: 40px auto; padding: 0 20px; }
        .grid { display: grid; grid-template-columns: 1fr 380px; gap: 30px; }
        .video-card {
            background: #1a1a2e;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #2a2a4e;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        .video-card img {
            width: 100%;
            display: block;
            border-radius: 0;
        }
        .video-label {
            padding: 14px 20px;
            background: #16213e;
            font-size: 0.9rem;
            color: #8888aa;
            border-top: 1px solid #2a2a4e;
        }
        .sidebar { display: flex; flex-direction: column; gap: 20px; }
        .card {
            background: #1a1a2e;
            border-radius: 16px;
            padding: 24px;
            border: 1px solid #2a2a4e;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }
        .card h2 { font-size: 1rem; color: #e94560; margin-bottom: 16px; display:flex; align-items:center; gap:8px; }
        .upload-area {
            border: 2px dashed #3a3a5e;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            transition: border-color 0.2s;
            cursor: pointer;
        }
        .upload-area:hover { border-color: #e94560; }
        .upload-area .icon { font-size: 2.5rem; margin-bottom: 10px; }
        .upload-area p { font-size: 0.85rem; color: #8888aa; margin-bottom: 14px; }
        input[type=file] { display: none; }
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #e94560, #c0392b);
            color: white;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            transition: opacity 0.2s, transform 0.1s;
            width: 100%;
            margin-top: 10px;
        }
        .btn:hover { opacity: 0.9; transform: translateY(-1px); }
        .btn:active { transform: translateY(0); }
        .status { margin-top: 12px; font-size: 0.85rem; color: #8888aa; text-align: center; }
        .status.active { color: #2ecc71; }
        .status.error { color: #e94560; }
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            background: #2ecc71;
            color: #000;
        }
        .badge.warning { background: #f39c12; }
        .badge.error { background: #e74c3c; }
        .info-row { display:flex; justify-content:space-between; align-items:center; padding: 8px 0; border-bottom: 1px solid #2a2a4e; font-size: 0.85rem; }
        .info-row:last-child { border-bottom: none; }
        .info-key { color: #8888aa; }
        #filename-display { margin-top:8px; font-size:0.8rem; color:#aaa; text-align:center; }
        progress { width:100%; height:8px; border-radius:4px; margin-top:10px; display:none; }
    </style>
</head>
<body>
<header>
    <span>🚗</span>
    <h1>Illegal Parking Detection System</h1>
</header>
<div class="container">
    <div class="grid">
        <div class="video-card">
            <img src="/video" alt="Live detection feed" id="feed">
            <div class="video-label">
                {% if video_name %}
                    ▶️ Analyzing: <strong>{{ video_name }}</strong>
                {% else %}
                    ⏸️ No video loaded — upload a video to begin
                {% endif %}
            </div>
        </div>
        <div class="sidebar">
            <div class="card">
                <h2>📤 Upload Video</h2>
                <form id="uploadForm" action="/upload" method="post" enctype="multipart/form-data">
                    <label for="file-input">
                        <div class="upload-area" id="drop-area">
                            <div class="icon">🎥</div>
                            <p>Click to select a video file (.mp4, .avi, .mov)</p>
                            <span class="btn" onclick="document.getElementById('file-input').click(); return false;">Choose File</span>
                        </div>
                    </label>
                    <input type="file" id="file-input" name="video" accept="video/*"
                           onchange="document.getElementById('filename-display').textContent = this.files[0]?.name || ''">
                    <div id="filename-display"></div>
                    <progress id="upload-progress" max="100"></progress>
                    <button type="submit" class="btn" style="margin-top:14px;">🚀 Start Analysis</button>
                </form>
                <div class="status {% if video_name %}active{% endif %}" id="status-msg">
                    {% if video_name %}✅ Currently analyzing: {{ video_name }}{% else %}Waiting for video upload...{% endif %}
                </div>
            </div>
            <div class="card">
                <h2>⚙️ Settings</h2>
                <div class="info-row">
                    <span class="info-key">Alert Threshold</span>
                    <span><strong>{{ time_threshold }}s</strong></span>
                </div>
                <div class="info-row">
                    <span class="info-key">Save Evidence</span>
                    <span class="badge {% if save_evidence %}{% else %}warning{% endif %}">
                        {% if save_evidence %}ON{% else %}OFF{% endif %}
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-key">Model</span>
                    <span class="badge">YOLOv8</span>
                </div>
            </div>
        </div>
    </div>
</div>
<script>
    // Upload via XHR for progress tracking
    document.getElementById('uploadForm').addEventListener('submit', function(e) {
        e.preventDefault();
        const form = this;
        const formData = new FormData(form);
        const xhr = new XMLHttpRequest();
        const progress = document.getElementById('upload-progress');
        const status = document.getElementById('status-msg');

        progress.style.display = 'block';
        status.textContent = 'Uploading...';
        status.className = 'status';

        xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                progress.value = pct;
                status.textContent = 'Uploading... ' + pct + '%';
            }
        });
        xhr.addEventListener('load', function() {
            if (xhr.status === 200 || xhr.status === 302) {
                window.location.reload();
            } else {
                status.textContent = '❌ Upload failed. Try again.';
                status.className = 'status error';
            }
        });
        xhr.open('POST', '/upload');
        xhr.send(formData);
    });
</script>
</body>
</html>
"""


@app.route('/')
def index():
    video_name = os.path.basename(current_video_path) if current_video_path else None
    return render_template_string(HTML_PAGE,
                                   video_name=video_name,
                                   time_threshold=TIME_THRESHOLD,
                                   save_evidence=SAVE_EVIDENCE)


@app.route('/upload', methods=['POST'])
def upload():
    global current_video_path
    if 'video' not in request.files:
        return "No file part", 400

    file = request.files['video']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    with video_lock:
        current_video_path = save_path

    print(f"✅ Video uploaded: {save_path}")
    return redirect(url_for('index'))


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)