import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T

# Define class names manually
class_names = ["good", "poor"]

# Load scripted model
model = torch.jit.load("quality_best.torchscript").eval()

# Image transform
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])

def run_detection(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        preds = model(img_tensor)[0]

    # Extract class with highest confidence (assumes preds[..., 4] is conf)
    if preds.shape[0] > 0:
        confidences = preds[:, 4]
        best_idx = torch.argmax(confidences).item()
        best_pred = preds[best_idx]
        class_id = int(best_pred[5])
        label = class_names[class_id]
    else:
        label = "unknown"

    # Save original image as output
    image.save(output_path)

    return label
