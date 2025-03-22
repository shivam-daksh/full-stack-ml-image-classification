from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import io
import base64
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the YOLO model
try:
    model = YOLO("yolov8n.pt")  # Replace with your YOLO model path or name
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    raise RuntimeError("Failed to load YOLO model. Ensure the model file is correct.")

# Helper function to encode image to base64
def encode_image_to_base64(image: Image.Image) -> str:
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise RuntimeError("Failed to encode image to base64.")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # Convert image to numpy array for YOLO
        image_array = np.array(image)

        # Make prediction
        results = model.predict(image_array)
        detections = results[0].boxes.data.cpu().numpy()  # Extract detections

        # Draw bounding boxes and labels on the image
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)  # Load custom font with size 20
        except IOError:
            font = ImageFont.load_default()  # Fallback to default font if custom font is unavailable

        formatted_results = []
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            class_id = int(detection[5])
            confidence = float(detection[4])  # Convert to Python float
            class_name = model.names[class_id]

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Add label with class name and confidence
            label = f"{class_name} {confidence:.2f}%"
            text_bbox = font.getbbox(label)  # Get bounding box of the text
            text_width = text_bbox[2] - text_bbox[0]  # Calculate text width
            text_height = text_bbox[3] - text_bbox[1]  # Calculate text height
            text_background = [x1, y1 - text_height, x1 + text_width, y1]  # Background for text
            draw.rectangle(text_background, fill="red")  # Draw background rectangle
            draw.text((x1, y1 - text_height), label, fill="white", font=font)  # Draw text

            # Append to results
            formatted_results.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidence),  # Convert to Python float
                "bbox": [float(x1), float(y1), float(x2), float(y2)]  # Convert bbox values to Python float
            })

        # Return predictions and processed image
        return {
            "predictions": formatted_results,
            "processed_image": f"data:image/png;base64,{encode_image_to_base64(image)}"
        }

    except HTTPException as http_err:
        logger.error(f"HTTP error: {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")