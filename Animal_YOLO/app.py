from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Load trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Home Page
@app.route("/")
def index():
    return render_template("index.html")

# Image Upload Detection
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        results = model(filepath)
        results[0].save(filename=filepath)

        return render_template("index.html", image=filepath)

    return "No File Uploaded"


# Live Camera Detection
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)