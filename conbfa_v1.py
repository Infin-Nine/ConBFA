import cv2
import numpy as np
import time
import ctypes
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# ================= CONFIGURATION =================
CAMERA_ID = 0
WINDOW_NAME = "Secure Face System"

# Model Selection: "SFace" is lightweight and accurate
MODEL_NAME = "SFace" 
DISTANCE_METRIC = "cosine"

# Tuning Parameters
ENROLL_FRAMES = 10           # How many frames to capture per angle
ANGLES = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]

# Authentication Thresholds (Lower is stricter)
# For SFace + Cosine: 0.4 to 0.5 is a good balance. 
# < 0.4 is very strict, > 0.6 is loose.
AUTH_THRESHOLD = 0.55        

BLUR_THRESHOLD = 60          # Skip blurry frames
LOCK_TIMEOUT = 3.0           # Seconds before locking if face is unknown

# ================= SYSTEM LOCK =================
def lock_windows():
    """Locks the Windows workstation."""
    print("🔒 LOCKING SYSTEM...")
    ctypes.windll.user32.LockWorkStation()

# ================= MEDIAPIPE SETUP =================
# Initialize MediaPipe Face Landmarker
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# UPDATE THIS PATH TO YOUR MODEL FILE
MODEL_PATH = "D:\\AI_ML_Workstation\\Projects\\Computer Vision\\Models\\face_landmarker.task"

try:
    with open(MODEL_PATH, 'r'): pass
except IOError:
    print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
    exit()

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ================= STATE VARIABLES =================
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Enrollment Data
enroll_data = {angle: [] for angle in ANGLES}
reference_embeddings = {}
current_enroll_angle = "CENTER"
is_enrolled = False

# Authentication State
score_history = deque(maxlen=10)  # Stores last 10 scores for smoothing
unknown_frame_count = 0
last_face_time = time.time()
windows_locked = False

print("\n📌 SYSTEM STARTED")
print(f"👉 Model: {MODEL_NAME}")
print("👉 Controls: 'C' (Center), 'L' (Left), 'R' (Right), 'U' (Up), 'D' (Down)")
print("👉 Press 'Esc' to quit.\n")

# ================= HELPER FUNCTIONS =================

def get_cosine_distance(embedding1, embedding2):
    """Calculates cosine distance between two embeddings."""
    a = embedding1 / np.linalg.norm(embedding1)
    b = embedding2 / np.linalg.norm(embedding2)
    return 1 - np.dot(a, b)

def get_embedding(face_image):
    """Generates face embedding using DeepFace."""
    try:
        # 'detector_backend="skip"' is CRITICAL for speed because
        # we already cropped the face using MediaPipe.
        result = DeepFace.represent(
            img_path=face_image,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend="skip"
        )
        return np.array(result[0]["embedding"])
    except:
        return None

def crop_face_from_landmarks(frame, landmarks):
    """Crops the face from the frame with some padding."""
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]

    # Add padding to ensure the whole face is captured
    pad = 20
    x1 = max(min(xs) - pad, 0)
    x2 = min(max(xs) + pad, w)
    y1 = max(min(ys) - pad, 0)
    y2 = min(max(ys) + pad, h)

    # Return empty if crop is invalid
    if x2 - x1 < 50 or y2 - y1 < 50:
        return np.array([])
        
    return frame[y1:y2, x1:x2]

def check_blur(image):
    """Returns True if image is too blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < BLUR_THRESHOLD

# ================= MAIN LOOP =================
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    frame_idx += 1
    
    # Run heavy recognition logic only every 3rd frame to reduce lag
    process_frame = (frame_idx % 3 == 0)

    # --- MEDIAPIPE DETECTION ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    status_text = "Initializing..."
    status_color = (255, 255, 255)

    if detection_result.face_landmarks:
        last_face_time = time.time()
        landmarks = detection_result.face_landmarks[0]
        face_crop = crop_face_from_landmarks(frame, landmarks)

        if face_crop.size > 0:
            if check_blur(face_crop):
                status_text = "⚠️ IMAGE BLURRY"
                status_color = (0, 255, 255) # Yellow
            else:
                
                # ================= ENROLLMENT MODE =================
                if not is_enrolled:
                    status_text = f"ENROLLING: {current_enroll_angle}"
                    status_color = (255, 165, 0) # Orange

                    # Only process embedding every few frames
                    if process_frame:
                        emb = get_embedding(face_crop)
                        if emb is not None:
                            # Save embedding if we need more frames for this angle
                            if len(enroll_data[current_enroll_angle]) < ENROLL_FRAMES:
                                enroll_data[current_enroll_angle].append(emb)

                            # Check if all angles are done
                            total_done_angles = sum(1 for a in ANGLES if len(enroll_data[a]) >= ENROLL_FRAMES)
                            
                            if total_done_angles == len(ANGLES):
                                # Calculate Average Embeddings
                                print("✅ Calculating Reference Models...")
                                for angle in ANGLES:
                                    reference_embeddings[angle] = np.mean(enroll_data[angle], axis=0)
                                is_enrolled = True
                                print("🎉 Enrollment Complete!")

                    # Show Frame Count on Screen
                    current_count = len(enroll_data[current_enroll_angle])
                    cv2.putText(frame, f"Frames: {current_count}/{ENROLL_FRAMES}", (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ================= AUTHENTICATION MODE =================
                else:
                    if process_frame:
                        emb = get_embedding(face_crop)
                        if emb is not None:
                            # Compare current face with all reference angles
                            distances = [get_cosine_distance(ref, emb) for ref in reference_embeddings.values()]
                            best_distance = min(distances)
                            
                            # Add to history for smoothing
                            score_history.append(best_distance)

                    # Calculate Rolling Average (This is the "Stability" logic)
                    if len(score_history) > 0:
                        avg_score = sum(score_history) / len(score_history)
                    else:
                        avg_score = 1.0 # Default high distance

                    # --- DECISION ---
                    if avg_score < AUTH_THRESHOLD:
                        status_text = f"🔓 ACCESS GRANTED ({avg_score:.2f})"
                        status_color = (0, 255, 0) # Green
                        unknown_frame_count = 0 # Reset fail counter
                    else:
                        status_text = f"🚫 UNAUTHORIZED ({avg_score:.2f})"
                        status_color = (0, 0, 255) # Red
                        unknown_frame_count += 1

                        # Lock if unauthorized for approx 1.5 seconds (running at 30fps/3 = 10 updates per sec)
                        if unknown_frame_count > 15:
                            status_text = "🔒 LOCKING..."
                            if not windows_locked:
                                windows_locked = True
                                lock_windows()
                                break
    else:
        # No Face Detected
        status_text = "NO FACE DETECTED"
        status_color = (100, 100, 100) # Gray
        
        # Lock if no face for too long
        if is_enrolled and (time.time() - last_face_time > LOCK_TIMEOUT):
            status_text = "🔒 TIMEOUT LOCK"
            if not windows_locked:
                windows_locked = True
                lock_windows()
                break

    # ================= DRAW UI =================
    # Status Bar Background
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow(WINDOW_NAME, frame)

    # ================= CONTROLS =================
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break # ESC
    # Angle switching
    if key == ord('c'): current_enroll_angle = "CENTER"
    if key == ord('l'): current_enroll_angle = "LEFT"
    if key == ord('r'): current_enroll_angle = "RIGHT"
    if key == ord('u'): current_enroll_angle = "UP"
    if key == ord('d'): current_enroll_angle = "DOWN"

cap.release()
cv2.destroyAllWindows()