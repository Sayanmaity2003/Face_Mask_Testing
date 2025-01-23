import cv2
import os
import numpy as np
import pygame
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from threading import Thread, Lock

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained mask detection model
model = load_model('mask_detector.h5')

# Initialize pygame mixer for alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')

# Global variables
video_active = False
alarm_active = False
lock = Lock()

# Set up video capture (initially no camera)
cap = cv2.VideoCapture(0)

# Function to detect face and mask
def detect_and_predict_mask(frame):
    """Detect faces and predict if they're wearing a mask."""
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces_found = []

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0)
        face = face / 255.0  # Normalize the image

        # Predict mask/no-mask
        (mask, withoutMask) = model.predict(face)[0]

        # Label the face with mask/no-mask
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        faces_found.append(label)

    return frame, faces_found

# Function to play alarm sound in a separate thread
def play_alarm():
    pygame.mixer.music.play(-1)  # Loop the alarm indefinitely

# Route to render home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)
    image = request.files['image']
    if image.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    filename = secure_filename(image.filename)
    image_path = os.path.join('static/uploads', filename)
    image.save(image_path)

    # Process the image for mask detection
    img = cv2.imread(image_path)
    img, results = detect_and_predict_mask(img)

    # Save processed image
    processed_image_path = os.path.join('static/uploads', 'processed_' + filename)
    cv2.imwrite(processed_image_path, img)

    # Encode image to base64
    with open(processed_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return render_template('index.html', result_image=encoded_image, results=results)

# Route to start video feed
@app.route('/start_video')
def start_video():
    global video_active
    with lock:
        video_active = True

    # Check if the camera is accessible
    if not cap.isOpened():
        return "Camera error: Could not access the webcam."

    return redirect(url_for('index'))

# Route to stop video feed
@app.route('/stop_video')
def stop_video():
    global video_active
    with lock:
        video_active = False

    # Release the camera and stop alarm
    cap.release()
    pygame.mixer.music.stop()  # Stop alarm sound
    return redirect(url_for('index'))

# Route to handle video feed (for streaming)
@app.route('/video_feed')
def video_feed():
    global cap
    if not video_active:
        return redirect(url_for('index'))

    ret, frame = cap.read()
    if not ret:
        return "Error: Failed to capture video"

    frame, faces = detect_and_predict_mask(frame)
    ret, jpeg = cv2.imencode('.jpg', frame)

    return Response(jpeg.tobytes(), mimetype='image/jpeg')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
