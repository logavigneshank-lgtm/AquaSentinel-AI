
from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
RESULT_FOLDER = os.path.join(os.path.dirname(__file__), "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    detection = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            filename = str(uuid.uuid4()) + ".jpg"
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)

            results = model.predict(
                source=input_path,
                conf=0.25,
                device="cpu"
            )

            result = results[0]

            if len(result.boxes) > 0:
                best_conf = 0
                best_class = "Unknown"

                for box in result.boxes:
                    conf = float(box.conf[0])

                    if conf > best_conf:
                        best_conf = conf
                        cls_id = int(box.cls[0])
                        best_class = model.names[cls_id]

                detection = best_class
                confidence = round(best_conf * 100, 2)
            else:
                detection = "No Human Detected"
                confidence = 0

            output_name = "result_" + filename
            output_path = os.path.join(RESULT_FOLDER, output_name)

            result.save(filename=output_path)

            result_image = "/results/" + output_name

    return render_template(
        "index.html",
        result_image=result_image,
        detection=detection,
        confidence=confidence
    )

@app.route("/results/<filename>")
def results(filename):
    from flask import send_from_directory
    return send_from_directory(RESULT_FOLDER, filename)

@app.route("/health")
def health():
    return {"status": "AquaSentinel AI is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
