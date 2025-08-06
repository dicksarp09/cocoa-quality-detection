from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Use local Gemma model directory
MODEL_PATH = r"E:\kagglehub\models\google\gemma-3n\transformers\gemma-3n-e2b\2"

# Load tokenizer and model from local path
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
