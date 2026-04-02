from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import io
from typing import Dict, Any, Optional
import logging
from urllib.parse import urlparse

import httpx

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

REMOTE_IMAGE_TIMEOUT = 10.0
MAX_REMOTE_IMAGE_SIZE = 10 * 1024 * 1024

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


def validate_remote_image_url(image_url: str) -> None:
    """Ensure the remote image URL uses http/https."""
    parsed = urlparse(image_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="image_url must be a valid http or https URL")


async def load_image_from_upload(image: UploadFile) -> Image.Image:
    """Load and normalize an uploaded image file."""
    contents = await image.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")


async def load_image_from_url(image_url: str) -> Image.Image:
    """Download, validate, and load a remote image."""
    validate_remote_image_url(image_url)

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=REMOTE_IMAGE_TIMEOUT) as client:
            response = await client.get(image_url)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=408, detail="Timed out while downloading image_url") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image_url: upstream returned {exc.response.status_code}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail="Failed to download image_url") from exc

    if len(response.content) > MAX_REMOTE_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Remote image is too large")

    content_type = response.headers.get("content-type", "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image_url did not return an image resource")

    try:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Downloaded content is not a valid image") from exc

@app.post("/detect")
async def detect(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    threshold: float = Form(0.7),
) -> Dict[str, Any]:
    """
    Endpoint to detect NSFW content in an uploaded image.

    Parameters:
    - image (UploadFile): The uploaded image file to be analyzed.
    - image_url (str): A remote image URL to be analyzed.
    - threshold (float): The confidence threshold for determining NSFW content.

    Returns:
    - dict: A dictionary containing the classification scores, the top label with its confidence,
            and a boolean indicating if the content is NSFW.
    """
    try:
        if bool(image) == bool(image_url):
            raise HTTPException(
                status_code=400,
                detail="Provide exactly one of image or image_url",
            )

        if image is not None:
            img = await load_image_from_upload(image)
        else:
            img = await load_image_from_url(image_url)
        
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=400, detail="Unable to process image") from e
