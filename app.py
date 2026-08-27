from flask import Flask, render_template, request
import os
import uuid
import cv2
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load ONNX model
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


def detect_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None, "Invalid image", 0

    original = image.copy()

    # Resize to 640x640
    resized = cv2.resize(image, (640, 640))

    # BGR -> RGB
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # HWC -> CHW
    input_image = resized.transpose(2, 0, 1)

    # Normalize
    input_image = input_image.astype(np.float32) / 255.0

    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)

    # ONNX inference
    outputs = session.run(None, {
        input_name: input_image
    })

    predictions = outputs[0][0]

    best_conf = 0
    best_box = None

    # YOLO output = [x, y, w, h, confidence]
    for i in range(predictions.shape[1]):
        x = predictions[0, i]
        y = predictions[1, i]
        w = predictions[2, i]
        h = predictions[3, i]
        conf = predictions[4, i]

        if conf > best_conf:
            best_conf = float(conf)
            best_box = (x, y, w, h)

    if best_box is not None and best_conf >= 0.25:
        x, y, w, h = best_box

        # Convert 640 coordinates to original image coordinates
        scale_x = original.shape[1] / 640
        scale_y = original.shape[0] / 640

        x1 = int((x - w / 2) * scale_x)
        y1 = int((y - h / 2) * scale_y)
        x2 = int((x + w / 2) * scale_x)
        y2 = int((y + h / 2) * scale_y)

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original.shape[1], x2)
        y2 = min(original.shape[0], y2)

        # Draw detection
        cv2.rectangle(
            original,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"Human: {best_conf * 100:.2f}%"

        cv2.putText(
            original,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        detection = "Human Detected"
        confidence = round(best_conf * 100, 2)

    else:
        detection = "No Human Detected"
        confidence = 0

    return original, detection, confidence


@app.route("/", methods=["GET", "POST"])
def index():

    result_image = None
    detection = None
    confidence = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename:

            filename = str(uuid.uuid4()) + ".jpg"

            input_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            file.save(input_path)

            result, detection, confidence = detect_image(
                input_path
            )

            if result is not None:

                output_name = "result_" + filename
                output_path = os.path.join(
                    RESULT_FOLDER,
                    output_name
                )

                cv2.imwrite(output_path, result)

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

    return send_from_directory(
        RESULT_FOLDER,
        filename
    )


@app.route("/health")
def health():
    return {
        "status": "AquaSentinel AI is running"
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(
        host="0.0.0.0",
        port=port
    )
