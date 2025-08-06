# cocoa-quality-detection
# Cocoa Quality Classifier with AI Feedback (Offline-First, Multimodal)
This project combines YOLOv8 for image classification and Gemma 3n for natural language feedback—running entirely offline to support cocoa farmers in low-connectivity regions.

## Key Features
Offline-First: No internet needed. All models run locally.

Multimodal AI: Combines vision and language—detects cocoa quality and explains the result in natural language.

Built with Gemma 3n: Uses the powerful, privacy-preserving language model for on-device generation.

## User Interfaces:

Command Line Interface (CLI) for fast predictions

🌐 (Optional) Flask web demo for local interface (if functional)

## Quickstart (CLI Version)
1. Install dependencies
pip install torch torchvision transformers ultralytics

2. Project structure


├── quality_best.pt               # YOLOv8 model (cocoa classification)
├── quality_scripted.pt (opt.)   # TorchScript version if using scripted
├── gemma_model.py               # Loads Gemma + response function
├── cli_app.py                   # Main CLI entry point
├── requirements.txt             # (Optional)
└── demo_images/
    └── sample.jpg

3. Run CLI

python cli_app.py "demo_images/sample.jpg"
🧠 How It Works
Image Classification (YOLOv8):

Detects cocoa bean type (e.g., poor, good)

Text Generation (Gemma 3n):

Receives a prompt like:
“What does it mean if a cocoa bean is classified as good?”

Outputs AI explanation using natural language.


Example Output

Predicted class: poor

Gemma Response:

Poof cocoa beans indicate poor drying or improper storage, leading to fungal contamination. This affects flavor and safety.
Demo Video
Included separately as CLI walk-through
Explains model pipeline, shows sample output, and demonstrates offline setup.

Model Details
YOLOv8: Fine-tuned on annotated cocoa bean images (dry, good, moldy, broken)

Gemma 3n (2B): Loaded via Hugging Face Transformers locally

Impact & Use Case
This app empowers cocoa producers to:

Visually assess bean quality

Understand implications with local AI explanation

Operate without internet, ensuring functionality in remote farming areas

Future Improvements
Add local language support (Twi, Ewe)

Enable voice input and audio response

Extend to leaf disease detection and biodiversity tracking

