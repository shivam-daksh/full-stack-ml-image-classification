from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
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

# Load the MobileNet model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize image to 224x224 (MobileNetV2 input size)
    image = image.resize((224, 224))
    # Convert to array and preprocess for MobileNetV2
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)

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
    
    # Preprocess image
    processed_array = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    
    # Format predictions
    results = [
        {
            "class_name": class_name,
            "probability": float(score)
        }
        for (_, class_name, score) in decoded_predictions
    ]
    
    # Return predictions and processed image
    return {
        "predictions": results,
        "processed_image": f"data:image/png;base64,{encode_image_to_base64(image)}"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}