from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import threading

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load your pre-trained face mask detection model
model = tf.keras.models.load_model(os.path.join(app.root_path, "mask_detector.h5"))

# Global variables to control video feed
video_active = False
lock = threading.Lock()


def detect_mask(face):
    """Detect mask on a single face."""
    resized_face = cv2.resize(face, (224, 224))  # Resize to match the model input
    normalized_face = resized_face / 255.0  # Normalize the pixel values
    input_data = np.expand_dims(normalized_face, axis=0)
    prediction = model.predict(input_data)
    label = "Mask" if prediction[0][0] > 0.5 else "No Mask"
    confidence = prediction[0][0] if label == "Mask" else 1 - prediction[0][0]
    return label, confidence


@app.route('/')
def index():
    """Render the home page."""
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the image and predict
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        results = []
        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]
            label, confidence = detect_mask(face)
            results.append((label, confidence * 100))

        return render_template("index.html", uploaded_image=filename, results=results)
    return redirect(url_for('index'))


@app.route('/start_video')
def start_video():
    """Start the live video feed."""
    global video_active
    with lock:
        video_active = True
    return redirect(url_for('index'))


@app.route('/stop_video')
def stop_video():
    """Stop the live video feed."""
    global video_active
    with lock:
        video_active = False
    return redirect(url_for('index'))


def generate_frames():
    """Generate frames for live video feed."""
    global video_active
    cap = cv2.VideoCapture(0)
    while True:
        with lock:
            if not video_active:
                break

        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            label, confidence = detect_mask(face)

            # Set color and label based on mask detection
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Encode the frame to be sent to the client
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    """Route for live video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()
