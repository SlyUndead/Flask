import os
HOME = os.getcwd()
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import io
from ultralytics import YOLOv10
import supervision as sv
model = YOLOv10(f'{HOME}/best.pt')
# model = YOLO(f'{HOME}/Flask/data.yaml')  # Replace with the correct path

# Optionally, load weights if needed
# model.load_weights(f'{HOME}/Flask/last.pt')

# dataset = sv.DetectionDataset.from_yolo(
#     images_directory_path=f"{dataset_path}/valid/images",
#     annotations_directory_path=f"{dataset_path}/valid/labels",
#     data_yaml_path=f"{HOME}/data.yaml"
# )
# model = YOLOv10(f'./Flask/best.pt')
app = Flask(__name__)
CORS(app, resources={r"/coordinates": {"origins": "*"}})
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    # response.headers['Access-Control-Allow-Headers'] = 'X-Requested-With, Content-Type, Accept'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    return response
@app.route('/')
def hello_world():
    return "<p>hello world</p>"

@app.route('/coordinates', methods=['POST'])
def get_coordinates():
    # Get the image from the request
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    # Get the image from the request
    file = request.files['image']
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 400

    # Run the YOLO model on the image
    results = model(img)  # Predict on the uploaded image
    print(results)
    # Initialize list to hold detections
    annotations = []
    for detection in results[0].boxes:
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        # confidence = detection.conf[0].item()  # If you need confidence score, uncomment this
        class_id = int(detection.cls[0].item())
        class_name = results[0].names[class_id]

        # Create vertices list based on bounding box
        vertices = [
            {"x": x1, "y": y1},
            {"x": x2, "y": y1},
            {"x": x2, "y": y2},
            {"x": x1, "y": y2},
            {"x": x1, "y": y1}
        ]

        # Add annotation to the list
        annotation = {
            "label": class_name,
            "vertices": vertices
        }
        annotations.append(annotation)

    # Prepare the response
    result = {
        "annotations": annotations,
        "status": "OPEN"
    }
    return jsonify(result)

app.run(debug=False, host='0.0.0.0')
