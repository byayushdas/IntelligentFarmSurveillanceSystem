import cv2
import time
import threading
from flask import Flask, Response, render_template, request
from inference_sdk import InferenceHTTPClient

# ----------------------------
# Roboflow API settings
# ----------------------------
API_KEY = "P3HHgjZsNxgi0BSe2aqq"
MODELS = {
    "cattle": "cattle-detection-rv8bv/2",   # Updated cattle model
    "pests": "pests-ux2g8/3",               # Updated pests model
    "diseases": "plants-diseases-detection-and-classification/12"
}


CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",  # switched to serverless
    api_key=API_KEY
)

# ----------------------------
# Shared camera
# ----------------------------
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

latest_frame = None
lock = threading.Lock()
running = True

def camera_reader():
    global latest_frame
    while running:
        ok, frame = cap.read()
        if ok:
            with lock:
                latest_frame = frame
        else:
            time.sleep(0.01)

reader_thread = threading.Thread(target=camera_reader, daemon=True)
reader_thread.start()

# ----------------------------
# Inference generator
# ----------------------------
def gen_mjpeg(model_key):
    model_id = MODELS.get(model_key)
    if not model_id:
        raise ValueError(f"Unknown model: {model_key}")

    while True:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        # Save to temp file for API
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            result = CLIENT.infer(temp_path, model_id=model_id)
        except Exception as e:
            print(f"API error: {e}")
            time.sleep(0.1)
            continue

        # Draw predictions
        for pred in result.get("predictions", []):
            x, y = int(pred["x"]), int(pred["y"])
            w, h = int(pred["width"]), int(pred["height"])
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2

            label = pred["class"]
            conf = pred["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 170, 255), 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (60, 170, 255), -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

        # Encode JPEG
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

@app.route("/")
def index():
    model = request.args.get("model", "pests")  # default pests
    return render_template("index.html", model=model)

@app.route("/stream/<model_key>")
def stream(model_key):
    if model_key not in MODELS:
        return "Unknown model", 404
    return Response(gen_mjpeg(model_key),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/health")
def health():
    return {"ok": True}

# ----------------------------
# Cleanup
# ----------------------------
import atexit
@atexit.register
def cleanup():
    global running
    running = False
    time.sleep(0.05)
    try:
        cap.release()
    except:
        pass

if __name__ == "__main__":
    # Visit: http://127.0.0.1:5000/?model=diseases  for plant disease detection
    # Visit: http://127.0.0.1:5000/?model=pests     for pest detection
    # Visit: http://127.0.0.1:5000/?model=cattle    for cattle detection
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
