import argparse
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- Load YOLO model
yolo_model = YOLO("quality_best.pt")

# --- Load Gemma model and tokenizer
MODEL_PATH = r"E:\kagglehub\models\google\gemma-3n\transformers\gemma-3n-e2b\2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
gemma_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# --- Function to generate AI response
def generate_response(prompt):
    # Temporary placeholder instead of Gemma
    return f"(AI response placeholder for prompt: '{prompt}')"

# --- Main CLI Function
def main(image_path):
    # Run prediction on image
    results = yolo_model(image_path)
    class_id = int(results[0].boxes.cls[0])
    class_name = yolo_model.names[class_id]
    
    print(f"\n✅ Predicted class: {class_name}")

    # Send to Gemma
    prompt = f"What does it mean if a cocoa bean is classified as {class_name}?"
    response = generate_response(prompt)
    
    print("\n💬 Gemma Response:\n")
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cocoa Quality CLI")
    parser.add_argument("image_path", help="Path to cocoa bean image")
    args = parser.parse_args()
    main(args.image_path)
