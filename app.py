from flask import Flask, request, jsonify, render_template
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# Initialize Flask app
app = Flask(__name__)

# Model directory
model_dir = "./model/deepfake_vs_real_image_detection"

# Load model
model = ViTForImageClassification.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=True
)

# Load processor
processor = ViTImageProcessor.from_pretrained(model_dir)

print("✅ Model and Processor loaded successfully!")

# ---------------- ROUTES ---------------- #

# Welcome page
@app.route('/')
def welcome():
    return render_template('Home.html')

# Detection page
@app.route('/detect')
def detect():
    return render_template('index.html')

@app.route('/how-it-works')
def how():
    return render_template("how-it-works.html")

@app.route('/model-info')
def model_info():
    return render_template("model-info.html")


# ---------------- PREDICTION API ---------------- #

@app.route('/predict', methods=['POST'])
def predict():
    try:

        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        image = Image.open(file.stream).convert("RGB")

        # Preprocess
        inputs = processor(images=image, return_tensors="pt")

        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_class_id = logits.argmax(-1).item()

        predicted_label = model.config.id2label[predicted_class_id]

        confidence_score = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()

        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence_score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- RUN APP ---------------- #

if __name__ == '__main__':
    app.run(debug=True)


# for run ----> flask run