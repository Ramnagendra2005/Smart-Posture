from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, send_file
from flask_cors import CORS
import cv2
import mediapipe as mp
import math
import numpy as np
import time
import google.generativeai as genai
import matplotlib.pyplot as plt
import matplotlib
import io
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from authlib.integrations.flask_client import OAuth

# Set matplotlib to use a non-GUI backend
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])
app.secret_key = 'YOUR_SECRET_KEY'  # For session management

# -------------------------------
# Gemini Chat API Initialization
# -------------------------------
API_KEY = "AIzaSyCNRnfdNqE12JOEkscK-3gg_6FjAFq-hlk"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# -------------------------------
# Initialize MediaPipe components
# -------------------------------
mp_pose = mp.solutions.pose
pose_front = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# -------------------------------
# Auto-framing parameters
# -------------------------------
# Parameters for auto-framing
frame_size = (640, 480)  # Default camera resolution
auto_frame_margin = 100  # Margin around the person (pixels)
auto_frame_smoothing = 0.3  # Smoothing factor (0-1, lower = smoother)
auto_frame_enabled = True  # Flag to enable/disable auto-framing
current_frame_rect = None  # Current framing rectangle
target_frame_rect = None   # Target framing rectangle

# -------------------------------
# Global variables for tracking and feedback
# -------------------------------
notification_message = ""
current_feedback = []  # Global list to store the current feedback messages
blink_count = 0
start_time = time.time()
start_tracking = time.time()  # Global tracking start time for today's overview

# Default posture scores
posture_scores = {
    "headTilt": 80,
    "shoulderAlignment": 70,
    "spinalPosture": 85,
    "hipBalance": 60,
    "legPosition": 75,
    "overallScore": 90
}

# Mock history data for graphs
posture_history = []
session_history = []
blink_rate_history = []
components_history = []

# -------------------------------
# Utility functions
# -------------------------------
def format_response(text):
    """Format bot responses properly."""
    formatted_text = text.replace("\n", "<br>")  # Preserve line breaks
    return formatted_text

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def send_notification(message):
    """Update notification message."""
    global notification_message
    notification_message = message

def eye_aspect_ratio(landmarks, left_indices, right_indices):
    """Calculate eye aspect ratio for blink detection."""
    left_eye = [landmarks[i] for i in left_indices]
    right_eye = [landmarks[i] for i in right_indices]
    def eye_ratio(eye):
        return (math.dist((eye[1].x, eye[1].y), (eye[5].x, eye[5].y)) +
                math.dist((eye[2].x, eye[2].y), (eye[4].x, eye[4].y))) / (
                2.0 * math.dist((eye[0].x, eye[0].y), (eye[3].x, eye[3].y)))
    return (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2.0

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def check_proximity(landmarks):
    """Check if user is too close to the screen."""
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    if shoulder_width > 0.6:
        send_notification("Move Back! You are too close to the screen.")
    elif shoulder_width < 0.48:
        send_notification("You are Leaning. Sit straight!")

def analyze_posture(landmarks):
    """Analyze posture and provide feedback."""
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    torso_mid = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
    torso_length = calculate_distance((nose.x, nose.y), torso_mid)
    shoulder_deviation = abs(left_shoulder.y - right_shoulder.y)
    if shoulder_deviation > 0.05 * torso_length:
        send_notification("Straighten Your Shoulders!")

def update_scores_from_feedback(feedback):
    """Update posture scores based on feedback messages."""
    global posture_scores
    # Adjust scores based on detected feedback in a simple way.
    if any("Straighten your neck" in msg for msg in feedback):
        posture_scores["headTilt"] = 60
    if any("Align your shoulders" in msg for msg in feedback):
        posture_scores["shoulderAlignment"] = 65
    if any("Straighten Your Shoulders" in msg for msg in feedback):
        posture_scores["spinalPosture"] = 80
    if any("Put down your left shoulder" in msg for msg in feedback):
        posture_scores["hipBalance"] = 70
    if any("Put down your right shoulder" in msg for msg in feedback):
        posture_scores["legPosition"] = 70
    # For overall score, simply take the average of the others.
    posture_scores["overallScore"] = int(
        (posture_scores["headTilt"] +
         posture_scores["shoulderAlignment"] +
         posture_scores["spinalPosture"] +
         posture_scores["hipBalance"] +
         posture_scores["legPosition"]) / 5
    )

def initialize_mock_data():
    """Initialize mock data for graphs."""
    global posture_history, session_history, blink_rate_history, components_history
    
    # Generate last 7 days of data
    for i in range(7):
        date = (datetime.now() - timedelta(days=6-i)).strftime("%m/%d")
        posture_history.append({
            "date": date,
            "score": np.random.randint(60, 95)
        })
    
    # Generate session data
    for i in range(5):
        session_history.append({
            "session": f"Session {i+1}",
            "score": np.random.randint(65, 95),
            "corrections": np.random.randint(2, 15)
        })
    
    # Generate blink rate data over time
    for i in range(8):
        hour = f"{(9 + i*2) % 24:02d}:00"
        blink_rate_history.append({
            "time": hour,
            "blinks": np.random.randint(10, 25)
        })
    
    # Generate component history data
    today_components = {
        "headTilt": posture_scores["headTilt"],
        "shoulderAlignment": posture_scores["shoulderAlignment"],
        "spinalPosture": posture_scores["spinalPosture"],
        "hipBalance": posture_scores["hipBalance"], 
        "legPosition": posture_scores["legPosition"]
    }
    yesterday_components = {
        "headTilt": np.random.randint(65, 85),
        "shoulderAlignment": np.random.randint(60, 80),
        "spinalPosture": np.random.randint(70, 90),
        "hipBalance": np.random.randint(55, 75),
        "legPosition": np.random.randint(65, 85)
    }
    
    components_history = {
        "today": today_components,
        "yesterday": yesterday_components
    }

# -------------------------------
# Auto-framing functions
# -------------------------------


# -------------------------------
# Configuration parameters
# -------------------------------
auto_frame_enabled   = True
auto_frame_margin    = 50           # pixels around detected box
min_visibility       = 0.6          # only use landmarks above this
prediction_horizon   = 0.2          # seconds ahead to predict
dt                   = 1.0 / 30     # assuming 30 Hz processing
kalman_process_noise = 1e-2
kalman_measure_noise = 1e-1

# -------------------------------
# Kalman filter for smoothing bbox
# -------------------------------
class KalmanBoxFilter:
    def __init__(self):
        # state: [x, y, w, h, vx, vy, vw, vh]
        self.kf = cv2.KalmanFilter(8, 4)
        # measurement: [x, y, w, h]
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        # transition matrix with velocity integration
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i+4] = dt
        self.kf.transitionMatrix = F
        self.kf.processNoiseCov     = np.eye(8, dtype=np.float32) * kalman_process_noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * kalman_measure_noise
        self.initialized = False

    def correct(self, meas):
        """Update with measurement [x,y,w,h]."""
        z = np.array(meas, dtype=np.float32).reshape(4,1)
        if not self.initialized:
            # initialize state with first measurement
            self.kf.statePost[:4,0] = z[:,0]
            self.kf.statePost[4:,0] = 0
            self.initialized = True
        return self.kf.correct(z)

    def predict(self, horizon=0.0):
        """Predict ahead by horizon seconds (optional)."""
        if horizon > 0:
            # temporarily bump transition for horizon
            Fh = np.eye(8, dtype=np.float32)
            for i in range(4):
                Fh[i, i+4] = horizon
            self.kf.transitionMatrix = Fh
        p = self.kf.predict()
        # restore original transition
        for i in range(4):
            self.kf.transitionMatrix[i, i+4] = dt
        return p[:4,0].tolist()
    
kf_box = KalmanBoxFilter()
current_frame_rect = None

# -------------------------------
# Bounding-box extraction
# -------------------------------
def get_person_bounding_box(pose_landmarks, frame_shape):
    """Get bounding box around detected person based on high-confidence landmarks."""
    if not pose_landmarks:
        return None

    h, w = frame_shape[:2]
    pts = [(lm.x * w, lm.y * h) 
           for lm in pose_landmarks.landmark 
           if lm.visibility >= min_visibility]
    if not pts:
        return None

    xs, ys = zip(*pts)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # dynamic padding
    pad_x = (x2 - x1) * 0.25 + auto_frame_margin
    pad_y = (y2 - y1) * 0.25 + auto_frame_margin

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

# -------------------------------
# Auto-framing application
# -------------------------------
def apply_auto_framing(frame, pose_results):
    """
    Crop & center the detected person smoothly using Kalman filtering + prediction.
    Returns a new frame (resized back to original) if enabled, or the unmodified frame.
    """
    global current_frame_rect, kf_box

    h, w = frame.shape[:2]
    # 1) measurement & filter update
    if getattr(pose_results, "pose_landmarks", None):
        meas = get_person_bounding_box(pose_results.pose_landmarks, frame.shape)
        if meas:
            kf_box.correct(meas)
            x, y, bw, bh = kf_box.predict(horizon=prediction_horizon)
            # preserve aspect ratio
            aspect = w / h
            if bw / bh > aspect:
                nw, nh = bw, bw / aspect
            else:
                nw, nh = bh * aspect, bh
            cx, cy = x + bw / 2, y + bh / 2
            fx = np.clip(cx - nw/2, 0, w - nw)
            fy = np.clip(cy - nh/2, 0, h - nh)
            current_frame_rect = [int(fx), int(fy), int(nw), int(nh)]

    # 2) crop & resize if we have a valid rect
    if current_frame_rect and auto_frame_enabled:
        x, y, cw, ch = current_frame_rect
        x = int(np.clip(x, 0, w - cw))
        y = int(np.clip(y, 0, h - ch))
        cw, ch = int(min(cw, w - x)), int(min(ch, h - y))
        crop = frame[y:y+ch, x:x+cw]
        frame = (cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                 if (cw, ch) != (w, h) else crop)
        # subtle green indicator
        cv2.rectangle(frame, (5,5), (25,25), (0,255,0), 2)

    return frame

# -------------------------------
# Video generation function (Front Camera)
# -------------------------------
def generate_front_camera():
    global blink_count, start_time, notification_message, current_feedback
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Initialize feedback messages for this frame
        feedback_messages = []

        # Flip for a mirror view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Pose
        results = pose_front.process(rgb_frame)
        
        # Apply auto-framing (must be before drawing landmarks)
        frame = apply_auto_framing(frame, results)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Update RGB frame after framing
        
        if results.pose_landmarks:
            # Draw landmarks on the frame
            # mp.solutions.drawing_utils.draw_landmarks(
            #     frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            # Check posture and proximity
            check_proximity(landmarks)
            analyze_posture(landmarks)

            left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
            right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
            neck = (landmarks[mp_pose.PoseLandmark.NOSE].x,
                    landmarks[mp_pose.PoseLandmark.NOSE].y)
            left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
            right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)

            shoulder_width = calculate_distance(left_shoulder, right_shoulder)
            torso_length = calculate_distance(neck, ((left_hip[0] + right_hip[0]) / 2,
                                                     (left_hip[1] + right_hip[1]) / 2))
            ideal_neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
            neck_deviation = abs(neck[0] - ideal_neck_x)
            shoulder_deviation = abs(left_shoulder[1] - right_shoulder[1])

            neck_threshold = 0.1 * shoulder_width
            shoulder_threshold = 0.05 * torso_length

            if neck_deviation > neck_threshold:
                feedback_messages.append("Straighten your neck!")
                if neck[0] > ideal_neck_x:
                    feedback_messages.append("Move neck to the left.")
                else:
                    feedback_messages.append("Move neck to the right.")

            if shoulder_deviation > shoulder_threshold:
                feedback_messages.append("Align your shoulders!")
                if left_shoulder[1] > right_shoulder[1]:
                    feedback_messages.append("Put down your left shoulder.")
                else:
                    feedback_messages.append("Put down your right shoulder.")

        # Process face mesh for eye blink detection
        face_results = face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            landmarks_face = face_results.multi_face_landmarks[0].landmark
            ear = eye_aspect_ratio(landmarks_face, left_eye_indices, right_eye_indices)
            if ear < 0.2:
                blink_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                if blink_count < 15:
                    feedback_messages.append("Blink More! Your eye blink rate is low.")
                blink_count = 0
                start_time = time.time()

        if notification_message:
            feedback_messages.append(notification_message)
            notification_message = ""

        # Update the global feedback variable
        current_feedback = feedback_messages.copy()

        # Add auto-framing status indicator
        if auto_frame_enabled:
            cv2.putText(frame, "Auto-framing: ON", (frame.shape[1] - 200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode and stream the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# -------------------------------
# Chat endpoints
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    try:
        response = model.generate_content(user_message)
        if response and response.text:
            formatted_response = format_response(response.text)
        else:
            formatted_response = "I'm sorry, I couldn't understand that."
        return jsonify({"response": formatted_response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

# -------------------------------
# Video and Auto-framing endpoints
# -------------------------------
@app.route('/video_feed_front')
def video_feed_front():
    return Response(
        generate_front_camera(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/toggle_auto_frame', methods=['POST'])
def toggle_auto_frame():
    global auto_frame_enabled
    auto_frame_enabled = not auto_frame_enabled
    return jsonify({"auto_frame_enabled": auto_frame_enabled})

@app.route('/set_auto_frame_settings', methods=['POST'])
def set_auto_frame_settings():
    global auto_frame_margin, auto_frame_smoothing
    data = request.json
    if 'margin' in data:
        auto_frame_margin = int(data['margin'])
    if 'smoothing' in data:
        auto_frame_smoothing = float(data['smoothing'])
    return jsonify({
        "margin": auto_frame_margin,
        "smoothing": auto_frame_smoothing
    })

# -------------------------------
# Feedback and data endpoints
# -------------------------------
@app.route('/feedback')
def feedback():
    global current_feedback
    # Update posture scores based on the current feedback
    update_scores_from_feedback(current_feedback)
    return jsonify({"feedback": current_feedback})

@app.route('/get_scores')
def get_scores():
    # Send the posture scores
    return jsonify(posture_scores)

@app.route('/overview')
def overview():
    # Get the posture scores
    postureScore = posture_scores.get("overallScore", 0)

    # Calculate the time tracked since the beginning of the session
    tracked_seconds = time.time() - start_tracking
    hours = tracked_seconds / 3600
    timeTracked = f"{hours:.1f} hrs"

    # Count corrections based on feedback messages
    corrections = 0
    for msg in current_feedback:
        if any(kw in msg for kw in ["Straighten your neck", "Align your shoulders", "Move neck", "Put down"]):
            corrections += 1

    # Count breaks taken based on low blink feedback messages
    breaksTaken = 0
    for msg in current_feedback:
        if "Blink More" in msg:
            breaksTaken += 1

    return jsonify({
        "postureScore": postureScore,
        "timeTracked": timeTracked,
        "corrections": corrections,
        "breaksTaken": breaksTaken
    })

@app.route('/report/data')
def report_data():
    # Calculate the time tracked since the beginning of the session
    tracked_seconds = time.time() - start_tracking
    hours = tracked_seconds / 3600
    
    # Get the current posture score
    current_score = posture_scores.get("overallScore", 0)
    
    # Calculate a random score trend
    trend_options = ["improving", "declining", "stable"]
    score_trend = np.random.choice(trend_options, p=[0.4, 0.2, 0.4])
    
    # Generate random average score
    average_score = int(np.random.randint(current_score-10, current_score+10))
    if average_score > 100:
        average_score = 97
    
    return jsonify({
        "currentScore": current_score,
        "averageScore": average_score,
        "scoreTrend": score_trend,
        "timeTracked": f"{hours:.1f}",
        "totalSessions": len(session_history),
        "totalCorrections": sum(session["corrections"] for session in session_history),
        "totalBreaks": np.random.randint(3, 12),
        "headTiltScore": posture_scores["headTilt"],
        "shoulderAlignmentScore": posture_scores["shoulderAlignment"],
        "spinalPostureScore": posture_scores["spinalPosture"],
        "hipBalanceScore": posture_scores["hipBalance"],
        "legPositionScore": posture_scores["legPosition"],
    })

# -------------------------------
# Graph generation endpoints
# -------------------------------
@app.route('/graph/posture_over_time')
def posture_over_time_graph():
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    # Extract data
    dates = [entry['date'] for entry in posture_history]
    scores = [entry['score'] for entry in posture_history]
    
    # Create custom color map with gradient
    colors = [(0.2, 0.4, 0.8, 1.0), (0.6, 0.2, 0.7, 1.0)]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
    
    # Create the plot
    plt.figure(figsize=(10, 6), facecolor='#1f2937')
    ax = plt.subplot(111)
    ax.set_facecolor('#1f2937')
    
    # Plot line and points
    plt.plot(dates, scores, '-', color='#4c72b0', linewidth=3, alpha=0.7)
    plt.plot(dates, scores, 'o', color='#8c54a1', markersize=8)
    
    # Fill area below the line with gradient
    plt.fill_between(dates, scores, color='#4c72b0', alpha=0.2)
    
    # Add today's score with annotation
    plt.scatter(dates[-1], scores[-1], s=120, color='#10b981', zorder=5)
    plt.annotate(f'Today: {scores[-1]}', 
                xy=(dates[-1], scores[-1]), 
                xytext=(dates[-1], scores[-1]+15),
                ha='center',
                color='white',
                fontweight='bold')
    
    # Styling
    plt.title('Posture Score Trend (Last 7 Days)', fontsize=16, color='white', fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12, color='#9ca3af', labelpad=10)
    plt.ylabel('Score', fontsize=12, color='#9ca3af', labelpad=10)
    plt.ylim(min(scores)-10, 105)
    
    # Grid and spines
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add horizontal line for good score threshold
    plt.axhline(y=80, color='#10b981', linestyle='--', alpha=0.5)
    plt.text(dates[0], 82, 'Good Posture Threshold', color='#10b981', fontsize=10)
    
    plt.tight_layout()
    
    # Save to BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

@app.route('/graph/posture_components')
def posture_components_graph():
    # Set the style
    plt.style.use('dark_background')
    
    # Create the figure
    fig = plt.figure(figsize=(8, 8), facecolor='#1f2937')
    
    # Extract data
    categories = ['Head Tilt', 'Shoulder Alignment', 'Spinal Posture', 'Hip Balance', 'Leg Position']
    today_values = [
        components_history['today']['headTilt'],
        components_history['today']['shoulderAlignment'],
        components_history['today']['spinalPosture'],
        components_history['today']['hipBalance'],
        components_history['today']['legPosition']
    ]
    yesterday_values = [
        components_history['yesterday']['headTilt'],
        components_history['yesterday']['shoulderAlignment'],
        components_history['yesterday']['spinalPosture'],
        components_history['yesterday']['hipBalance'],
        components_history['yesterday']['legPosition']
    ]
    
    # Compute angles for the radar chart
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add values to complete the loop
    today_values += today_values[:1]
    yesterday_values += yesterday_values[:1]
    
    # Set up the radar chart
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('#1f2937')
    
    # Plot the data
    ax.plot(angles, today_values, 'o-', linewidth=2, color='#818cf8', label='Today')
    ax.fill(angles, today_values, color='#818cf8', alpha=0.25)
    
    ax.plot(angles, yesterday_values, 'o-', linewidth=2, color='#a78bfa', label='Yesterday')
    ax.fill(angles, yesterday_values, color='#a78bfa', alpha=0.1)
    
    # Customize the chart
    ax.set_theta_offset(np.pi / 2)  # Start from the top
    ax.set_theta_direction(-1)  # Go clockwise
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, color='white')
    
    # Set the y-axis limits and labels
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color='#9ca3af')
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False, fontsize=12)
    plt.title('Posture Component Analysis', fontsize=16, color='white', fontweight='bold', pad=20)
    
    # Add concentric circles and grid styling
    ax.grid(color='gray', alpha=0.3)
    
    plt.tight_layout()
    
    # Save to BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

@app.route('/graph/blink_rate')
@app.route('/graph/blink_rate')
def blink_rate_graph():
    plt.style.use('dark_background')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1f2937')
    ax.set_facecolor('#1f2937')
    
    # Extract data
    times = [entry['time'] for entry in blink_rate_history]
    blinks = [entry['blinks'] for entry in blink_rate_history]
    
    # Create the bar plot with gradient colors
    bars = ax.bar(times, blinks, color='#818cf8', alpha=0.8, width=0.7)
    
    # Add a subtle gradient effect to bars
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.cool(i/len(bars)))
    
    # Add value labels on top of each bar
    for i, v in enumerate(blinks):
        ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=10, color='white')
    
    # Styling
    ax.set_title('Blink Rate Throughout the Day', fontsize=16, color='white', fontweight='bold', pad=20)
    ax.set_xlabel('Time of Day', fontsize=12, color='#9ca3af', labelpad=10)
    ax.set_ylabel('Blinks per Minute', fontsize=12, color='#9ca3af', labelpad=10)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add reference line for healthy blink rate
    ax.axhline(y=15, color='#10b981', linestyle='--', alpha=0.7)
    ax.text(0, 16, 'Healthy Blink Rate (15 bpm)', color='#10b981', fontsize=10)
    
    # Add reference line for low blink rate
    ax.axhline(y=10, color='#ef4444', linestyle='--', alpha=0.7)
    ax.text(len(times)-1, 8, 'Low Blink Rate Warning', color='#ef4444', fontsize=10, ha='right')
    
    plt.tight_layout()
    
    # Save to BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

@app.route('/graph/session_history')
def session_history_graph():
    plt.style.use('dark_background')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1f2937')
    ax.set_facecolor('#1f2937')
    
    # Extract data
    sessions = [entry['session'] for entry in session_history]
    scores = [entry['score'] for entry in session_history]
    corrections = [entry['corrections'] for entry in session_history]
    
    # Create a twin axis for corrections
    ax2 = ax.twinx()
    
    # Plot the data
    score_bars = ax.bar([i-0.2 for i in range(len(sessions))], scores, width=0.4, color='#8b5cf6', alpha=0.8, label='Score')
    correction_bars = ax2.bar([i+0.2 for i in range(len(sessions))], corrections, width=0.4, color='#ec4899', alpha=0.8, label='Corrections')
    
    # Add value labels on top of each bar
    for i, v in enumerate(scores):
        ax.text(i-0.2, v + 3, str(v), ha='center', fontsize=10, color='white')
    
    for i, v in enumerate(corrections):
        ax2.text(i+0.2, v + 0.5, str(v), ha='center', fontsize=10, color='white')
    
    # Styling
    ax.set_title('Session History Analysis', fontsize=16, color='white', fontweight='bold', pad=20)
    ax.set_xlabel('Session', fontsize=12, color='#9ca3af', labelpad=10)
    ax.set_ylabel('Score', fontsize=12, color='#8b5cf6', labelpad=10)
    ax2.set_ylabel('Corrections', fontsize=12, color='#ec4899', labelpad=10)
    
    # Set x-ticks
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels(sessions)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)
    
    plt.tight_layout()
    
    # Save to BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

@app.route('/graph/heatmap')
def posture_heatmap():
    plt.style.use('dark_background')
    
    # Create sample data for the heatmap
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hours = ["9AM", "10AM", "11AM", "12PM", "1PM", "2PM", "3PM", "4PM", "5PM"]
    
    # Generate random data with some patterns
    np.random.seed(42)  # for reproducibility
    data = 60 + np.random.randint(0, 40, size=(len(hours), len(days)))
    
    # Make working hours have better posture
    for i in range(2, 6):  # 11AM to 3PM
        for j in range(5):  # Monday to Friday
            data[i, j] += 15
            if data[i, j] > 100:
                data[i, j] = 100
    
    # Make evening hours worse
    for i in range(7, 9):  # 4PM to 5PM
        for j in range(5):  # Monday to Friday
            data[i, j] -= 10
            if data[i, j] < 50:
                data[i, j] = 50
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1f2937')
    
    # Create a custom colormap from red to green
    cmap = LinearSegmentedColormap.from_list('posture_cmap', ['#ef4444', '#eab308', '#10b981'], N=100)
    
    # Create the heatmap
    sns.heatmap(data, ax=ax, cmap=cmap, annot=True, fmt="d", cbar_kws={'label': 'Posture Score'})
    
    # Styling
    ax.set_title('Posture Score Heatmap (Weekly Pattern)', fontsize=16, color='white', fontweight='bold', pad=20)
    ax.set_xlabel('Day of Week', fontsize=12, color='#9ca3af', labelpad=10)
    ax.set_ylabel('Hour of Day', fontsize=12, color='#9ca3af', labelpad=10)
    
    # Set axis labels
    ax.set_xticklabels(days)
    ax.set_yticklabels(hours)
    
    plt.tight_layout()
    
    # Save to BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

# -------------------------------
# OAuth Configuration for Google Login
# -------------------------------
app.secret_key = 'YOUR_SECRET_KEY'
from authlib.integrations.flask_client import OAuth
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='42043198999-25lgpoq8daebi2ks4khk1bhkcjqo41dg.apps.googleusercontent.com',
    client_secret='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',
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    api_base_url='https://www.googleapis.com/oauth2/v2/',
    client_kwargs={'scope': 'openid email profile'},
)

@app.route('/login')
def login():
    # The _external parameter is required to build an absolute redirect URL.
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    # Exchange the authorization code for a token
    token = google.authorize_access_token()
    # Fetch the user’s profile information from Google
    resp = google.get('userinfo', token=token)
    user_info = resp.json()
    # Store the user info in session
    session['user'] = user_info
    # After successful login, redirect to the frontend app (adjust URL if needed)
    return redirect('http://localhost:5173')

@app.route('/api/user')
def user():
    # Return the logged-in user’s info as JSON
    user_info = session.get('user')
    if user_info:
        return jsonify(user_info)
    return jsonify({'error': 'Not logged in'}), 401

@app.route('/logout')
def logout():
    session.pop('user', None)
    # Redirect back to the frontend landing page
    return redirect('http://localhost:5173')
# -------------------------------
# Protected routes (corrected redirects)
# -------------------------------
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/reports')
def reports():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('reports.html')

@app.route('/settings')
def settings():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('settings.html')



@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify({
        "auto_frame_enabled": auto_frame_enabled,
        "auto_frame_margin": auto_frame_margin,
        "auto_frame_smoothing": auto_frame_smoothing
    })

if __name__ == "__main__":
    initialize_mock_data()
    app.run(debug=True, threaded=True)


