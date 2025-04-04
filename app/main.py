from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import io
from typing import Dict, Any
import logging

app = FastAPI()

# Enable CORS for all origins and methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    Root endpoint that returns a simple message indicating the API is deployed.
    """
    return {"message": "✅ NSFW Image Detection API is deployed and running"}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model and processor initialization
model_id = "Falconsai/nsfw_image_detection"

# Set number of threads for CPU optimization
torch.set_num_threads(8)

# Load model with optimizations
def load_optimized_model():
    logger.info("Loading model...")
    model = AutoModelForImageClassification.from_pretrained(model_id)
    
    # Enable inference optimizations
    model.eval()
    return model

# Initialize model and processor
logger.info("Starting model initialization...")
model = load_optimized_model()
processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
logger.info("Model and processor loaded successfully")

@torch.no_grad()  # Disable gradient calculation during inference
def classify_image(image: Image.Image) -> Dict[str, float]:
    """Classify the input image and return prediction scores."""
    try:
        # Resize image to model's expected size before processing
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Process image in the correct format
        inputs = processor(images=image, return_tensors="pt")
        
        # Run inference
        outputs = model(inputs.pixel_values)
        probabilities = outputs.logits.softmax(dim=1)[0]
        
        # Convert to Python floats
        labels = model.config.id2label
        return {labels[i]: float(probabilities[i]) for i in range(len(probabilities))}
    except Exception as e:
        logger.error(f"Error during image classification: {e}")
        raise

@app.post("/detect")
async def detect(image: UploadFile = File(...), threshold: float = Form(0.7)) -> Dict[str, Any]:
    """
    Endpoint to detect NSFW content in an uploaded image.

    Parameters:
    - image (UploadFile): The image file to be analyzed.
    - threshold (float): The confidence threshold for determining NSFW content.

    Returns:
    - dict: A dictionary containing the classification scores, the top label with its confidence,
            and a boolean indicating if the content is NSFW.
    """
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        scores = classify_image(img)
        
        # Determine if the content is NSFW based on the threshold
        nsfw_score = scores.get("nsfw", 0)
        is_nsfw = nsfw_score > threshold

        # Identify the top label and its confidence
        top_label = max(scores, key=scores.get)
        top_confidence = scores[top_label]

        return {
            "scores": scores,
            "top": {
                "label": top_label,
                "confidence": top_confidence
            },
            "nsfw": is_nsfw
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise