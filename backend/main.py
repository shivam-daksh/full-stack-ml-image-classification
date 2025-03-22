from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from PIL import Image
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
    print(detections)
    print()
    # Format predictions
    formatted_results = [
        {
            "class_id": int(detection[5]),
            "class_name": model.names[int(detection[5])],
            "confidence": float(detection[4]),
            "bbox": detection[:4].tolist()  # [x1, y1, x2, y2]
        }
        for detection in detections
    ]
    
    # Return predictions and processed image
    return {
        "predictions": formatted_results,
        "processed_image": f"data:image/png;base64,{encode_image_to_base64(image)}"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
