from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

app = FastAPI()

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Replace with your YOLO model path or name

def encode_image_to_base64(image: Image.Image) -> str:
    # Convert PIL Image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and validate image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Convert image to numpy array for YOLO
    image_array = np.array(image)
    
    # Make prediction
    results = model.predict(image_array)
    detections = results[0].boxes.data.cpu().numpy()  # Extract detections
    
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Load default font
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
        draw.text((2*x1, 2*(y1 - text_height)), label, fill="white", font=font)  # Draw text
        
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