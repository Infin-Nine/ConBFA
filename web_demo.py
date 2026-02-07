import cv2
import time
import threading
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, redirect, url_for
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# ================= CONFIGURATION =================
CAMERA_ID = 0  # Change to 0 or 1 based on your camera
MODEL_NAME = "SFace"  # Lightweight and Fast
DISTANCE_METRIC = "cosine"

# Enrollment Settings
ENROLL_FRAMES = 10
ANGLES = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]

# Security Thresholds
# SFace thresholds: Lower is stricter. 0.5 to 0.6 is standard.
AUTH_THRESHOLD = 0.55  
BLUR_THRESHOLD = 60
LOCK_TIMEOUT = 0.5  # Seconds to wait before locking if no face seen

# Optimization
PROCESS_EVERY_N_FRAMES = 3  # Process 1 out of every 3 frames

# ================= GLOBAL STATE =================
# We use a lock to ensure thread safety when writing/reading frames
lock = threading.Lock()

global_frame = None
auth_status = "LOCKED"       # LOCKED | UNLOCKED
current_angle = "CENTER"
is_enrolled = False

# Enrollment Progress Tracking
enroll_data = {a: [] for a in ANGLES}
enroll_progress = {a: 0 for a in ANGLES} # To show on UI
reference_embeddings = {}

# Authentication Stability
score_history = deque(maxlen=10) # Last 10 scores for smoothing
unknown_counter = 0
last_face_seen_time = time.time()

# ================= FLASK APP =================
app = Flask(__name__)

# HTML Template (Frontend)
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
<title>Secure Face System</title>
<style>
    body { background-color: #121212; color: white; font-family: 'Segoe UI', sans-serif; display: flex; height: 100vh; margin: 0; }
    #video-container { flex: 2; display: flex; justify-content: center; align-items: center; background: #000; }
    #video-feed { width: 90%; border: 2px solid #333; border-radius: 8px; }
    #controls { flex: 1; padding: 20px; background: #1e1e1e; border-left: 1px solid #333; display: flex; flex-direction: column; }
    
    .status-box { padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px; font-weight: bold; font-size: 24px; }
    .locked { background-color: #cf6679; color: #000; }
    .unlocked { background-color: #03dac6; color: #000; }
    
    h3 { border-bottom: 1px solid #444; padding-bottom: 10px; }
    
    .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    button { padding: 12px; border: none; border-radius: 4px; background: #333; color: white; cursor: pointer; font-size: 14px; transition: 0.2s; }
    button:hover { background: #555; }
    button.active { background: #bb86fc; color: black; font-weight: bold; }

    .progress-bar { font-size: 12px; color: #aaa; margin-top: 5px; }
    
    .secret-link { margin-top: auto; padding: 15px; background: #3700b3; color: white; text-align: center; text-decoration: none; border-radius: 5px; font-weight: bold; }
    .secret-link:hover { background: #6200ea; }
</style>
</head>
<body>

<div id="video-container">
  <img id="video-feed" src="/video_feed">
</div>

<div id="controls">
  <h2>🛡️ ConBFA Security</h2>
  
  <div id="status-indicator" class="status-box locked">LOCKED</div>
  <div id="enroll-status" style="text-align:center; margin-bottom:10px; color:#aaa;">System Initializing...</div>

  <h3>📝 Enrollment Controls</h3>
  <div class="btn-grid">
    <button onclick="setAngle('CENTER')" id="btn-CENTER">CENTER</button>
    <button onclick="setAngle('LEFT')" id="btn-LEFT">LEFT</button>
    <button onclick="setAngle('RIGHT')" id="btn-RIGHT">RIGHT</button>
    <button onclick="setAngle('UP')" id="btn-UP">UP</button>
    <button onclick="setAngle('DOWN')" id="btn-DOWN">DOWN</button>
  </div>
  <div id="angle-progress" class="progress-bar">Current Angle Progress: 0/10</div>

  <a href="/secret_dashboard" target="_blank" class="secret-link">🔐 Access Protected Resource</a>
</div>

<script>
    function setAngle(angle) {
        fetch('/set_angle/' + angle);
        document.querySelectorAll('button').forEach(b => b.classList.remove('active'));
        document.getElementById('btn-' + angle).classList.add('active');
    }

    setInterval(() => {
        fetch('/status').then(r => r.json()).then(data => {
            // Update Auth Status
            const box = document.getElementById('status-indicator');
            box.innerText = data.auth_state;
            box.className = "status-box " + (data.auth_state === "UNLOCKED" ? "unlocked" : "locked");
            
            // Update Enrollment Info
            document.getElementById('enroll-status').innerText = data.is_enrolled ? "✅ User Enrolled" : "⚠️ Enrollment Needed";
            document.getElementById('angle-progress').innerText = `Angle ${data.current_angle}: ${data.frames_captured}/${data.target_frames}`;
            
            // Highlight active button
            if(!data.is_enrolled) {
                document.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                document.getElementById('btn-' + data.current_angle).classList.add('active');
            }
        });
    }, 500);
</script>
</body>
</html>
"""

# ================= FLASK ROUTES =================

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with lock:
                if global_frame is None:
                    continue
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', global_frame)
                frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03) # Limit stream to ~30 FPS

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/status")
def status_endpoint():
    # API for frontend to poll current state
    return jsonify({
        "auth_state": auth_status,
        "is_enrolled": is_enrolled,
        "current_angle": current_angle,
        "frames_captured": enroll_progress[current_angle],
        "target_frames": ENROLL_FRAMES
    })

@app.route("/set_angle/<angle>")
def set_angle_endpoint(angle):
    global current_angle
    if angle in ANGLES:
        current_angle = angle
    return "OK"

# 🔥 PROTECTED RESOURCE LOGIC 🔥
@app.route("/secret_dashboard")
def protected_resource():
    """
    This route represents the sensitive data/file/page.
    It can ONLY be accessed if the user is authenticated.
    """
    if auth_status == "UNLOCKED":
        return """
        <div style='background:green; color:white; padding:50px; text-align:center; font-family:Arial;'>
            <h1>🎉 ACCESS GRANTED</h1>
            <p>Welcome to the Secret Dashboard.</p>
            <p>Here is your confidential data: <b>XYZ-123-SECRET</b></p>
        </div>
        """
    else:
        return """
        <div style='background:red; color:white; padding:50px; text-align:center; font-family:Arial;'>
            <h1>🚫 ACCESS DENIED</h1>
            <p>Face not recognized or System Locked.</p>
            <p>Please authenticate via the camera first.</p>
        </div>
        """, 403

# ================= CORE LOGIC (THREAD) =================

def extract_embedding(face_img):
    """Gets 128-d embedding using SFace."""
    try:
        # backend="skip" because we use MediaPipe for detection
        rep = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend="skip"
        )
        return np.array(rep[0]["embedding"])
    except:
        return None

def crop_face(frame, landmarks):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    
    pad = 20
    x1, x2 = max(min(xs) - pad, 0), min(max(xs) + pad, w)
    y1, y2 = max(min(ys) - pad, 0), min(max(ys) + pad, h)
    
    if x2-x1 < 40 or y2-y1 < 40: return np.array([])
    return frame[y1:y2, x1:x2]

def get_cosine_dist(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)

def check_blur(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < BLUR_THRESHOLD

def processing_loop():
    global global_frame, auth_status, is_enrolled, unknown_counter, last_face_seen_time, reference_embeddings
    
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # MediaPipe Setup
    BaseOptions = python.BaseOptions
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode
    
    # Update Path!
    model_path = "D:\\AI_ML_Workstation\\Projects\\Computer Vision\\Models\\face_landmarker.task"
    
    try:
        with open(model_path, 'r'): pass
    except:
        print("Model not found. Please check path.")
        return

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1
    )
    landmarker = FaceLandmarker.create_from_options(options)
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        frame_idx += 1
        
        # Determine if we run heavy logic this frame
        run_recognition = (frame_idx % PROCESS_EVERY_N_FRAMES == 0)
        
        # Convert for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, int(time.time()*1000))
        
        overlay_color = (0, 0, 255) # Default Red
        
        if result.face_landmarks:
            last_face_seen_time = time.time()
            landmarks = result.face_landmarks[0]
            face = crop_face(frame, landmarks)
            
            if face.size > 0:
                if not check_blur(face):
                    
                    # --- ENROLLMENT LOGIC ---
                    if not is_enrolled:
                        overlay_color = (255, 165, 0) # Orange
                        
                        if run_recognition:
                            emb = extract_embedding(face)
                            if emb is not None:
                                if len(enroll_data[current_angle]) < ENROLL_FRAMES:
                                    enroll_data[current_angle].append(emb)
                                    enroll_progress[current_angle] = len(enroll_data[current_angle])
                                
                                # Check Completion
                                if all(len(enroll_data[a]) >= ENROLL_FRAMES for a in ANGLES):
                                    print("Calculating averages...")
                                    for a in ANGLES:
                                        reference_embeddings[a] = np.mean(enroll_data[a], axis=0)
                                    is_enrolled = True
                                    print("Enrollment DONE.")

                    # --- AUTHENTICATION LOGIC ---
                    else:
                        if run_recognition:
                            emb = extract_embedding(face)
                            if emb is not None:
                                # Compare against all angles
                                dists = [get_cosine_dist(ref, emb) for ref in reference_embeddings.values()]
                                best_dist = min(dists)
                                
                                # Add to history (Rolling Average)
                                score_history.append(best_dist)
                                
                                avg_score = sum(score_history) / len(score_history)
                                
                                if avg_score < AUTH_THRESHOLD:
                                    auth_status = "UNLOCKED"
                                    unknown_counter = 0
                                    overlay_color = (0, 255, 0) # Green
                                else:
                                    unknown_counter += 1
                                    overlay_color = (0, 0, 255) # Red
                                    
                                    # Lock only after persistent failure (approx 1.5s)
                                    if unknown_counter > (15 / PROCESS_EVERY_N_FRAMES):
                                        auth_status = "LOCKED"
                        
                        # Maintain color based on status for skipped frames
                        if auth_status == "UNLOCKED": overlay_color = (0, 255, 0)

        else:
            # NO FACE
            if is_enrolled and (time.time() - last_face_seen_time > LOCK_TIMEOUT):
                auth_status = "LOCKED"
        
        # Draw basic status on frame for debug
        cv2.putText(frame, f"STATUS: {auth_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, overlay_color, 2)
        
        # Update global frame safely
        with lock:
            global_frame = frame.copy()

# ================= MAIN START =================
if __name__ == "__main__":
    # Start the face recognition loop in a separate thread
    t = threading.Thread(target=processing_loop)
    t.daemon = True
    t.start()
    
    print("🚀 Starting Web Server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)