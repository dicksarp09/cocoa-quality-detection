from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from detect import run_detection
from gemma_model import generate_response

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def run_yolo_and_generate_prompt(input_path):
    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    result = run_detection(input_path, output_path)

    # Optional: customize prompt with result if available
    prompt = "Describe this cocoa bean quality based on image analysis results."
    if result:
        prompt += f" Detected class: {result}."
    return prompt

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST" and "user_input" in request.form:
        user_input = request.form["user_input"]
        prediction = generate_response(user_input)
    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods=["POST"])
def predict():
    ai_report = None
    image_path = None
    uploaded = False
    if "file" in request.files:
        file = request.files["file"]
        if file and file.filename != "":
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filepath)
            image_path = filepath
            prompt = run_yolo_and_generate_prompt(filepath)
            ai_report = generate_response(prompt)
            uploaded = True
    return render_template("index.html", ai_report=ai_report, image_path=image_path, uploaded=uploaded)
