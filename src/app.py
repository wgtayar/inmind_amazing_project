from flask import Flask, request, jsonify, send_file
import onnxruntime as ort
import numpy as np
import cv2
import io
import os

app = Flask(__name__)

# Load your ONNX model
ort_session = ort.InferenceSession("/home/wgt/Desktop/Models/customResNet.onnx")

def preprocess_image(image):
    # Resize the image to 224x224 as done during training
    image = cv2.resize(image, (224, 224))
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    # HWC to CHW
    image = np.transpose(image, (2, 0, 1))
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def postprocess(boxes, scores, labels, score_threshold=0.5):
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    for box, score, label in zip(boxes, scores, labels):
        if score > score_threshold:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_labels.append(label)
    return filtered_boxes, filtered_scores, filtered_labels

@app.route('/models', methods=['GET'])
def list_models():
    # Simplified to return a single model name; extend as needed
    return jsonify(["customResNet.onnx"])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error='No file provided'), 400

    filestr = request.files['file'].read()
    # Convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # Convert numpy array to image
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    input_tensor = preprocess_image(image)
    # ONNX Runtime expects inputs as {input_name: tensor}
    inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outputs = ort_session.run(None, inputs)
    
    # Assuming model output is boxes, labels, and scores
    boxes, scores, labels = ort_outputs
    boxes, scores, labels = postprocess(boxes[0], scores[0], labels[0])
    
    return jsonify({'boxes': boxes, 'scores': scores, 'labels': labels})

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify(error='No file provided'), 400

    filestr = request.files['file'].read()
    npimg = np.fromstring(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    input_tensor = preprocess_image(image)
    inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outputs = ort_session.run(None, inputs)
    
    boxes, scores, labels = ort_outputs
    boxes, scores, labels = postprocess(boxes[0], scores[0], labels[0], score_threshold=0.5)
    
    for box in boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    
    # Save or send the image with boxes
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
    return send_file(io.BytesIO(image_bytes), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
