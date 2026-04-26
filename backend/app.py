from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import logging
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="JAMV MODEL - Diabetic Retinopathy API",
    description="ML Model for Diabetic Retinopathy Detection from retinal images"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best_model.keras")

# Try to load model with keras
try:
    from tensorflow import keras
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully!")
except ImportError:
    logger.warning("TensorFlow not available, using mock model")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# DR Severity classes
DR_CLASSES = [
    "No DR (No Diabetic Retinopathy)",
    "Mild NPDR (Non-Proliferative DR)",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR (PDR)"
]

class ImagePredictionResponse(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float
    all_probabilities: dict
    severity: str
    recommendation: str

def preprocess_image(image_bytes, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def get_severity_info(class_id):
    """Get severity level and recommendations"""
    severity_info = {
        0: {"severity": "None", "recommendation": "Continue regular annual eye exams. Maintain good blood sugar control."},
        1: {"severity": "Mild", "recommendation": "Schedule eye exam in 6-12 months. Monitor blood sugar and blood pressure."},
        2: {"severity": "Moderate", "recommendation": "Schedule eye exam within 3-6 months. Consult ophthalmologist for treatment options."},
        3: {"severity": "Severe", "recommendation": "Immediate consultation with ophthalmologist required. Laser treatment may be needed."},
        4: {"severity": "Proliferative", "recommendation": "URGENT: Immediate medical attention required. Vitrectomy and laser therapy may be needed to prevent vision loss."}
    }
    return severity_info.get(class_id, {"severity": "Unknown", "recommendation": "Consult your doctor"})

@app.get("/")
async def root():
    return {
        "message": "JAMV MODEL - Diabetic Retinopathy Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict_image": "/predict/image",
            "classes": DR_CLASSES
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "supported_classes": DR_CLASSES
    }

@app.get("/classes")
async def get_classes():
    return {"classes": DR_CLASSES, "description": "Diabetic Retinopathy severity classification"}

@app.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict diabetic retinopathy from retinal image"""
    if model is None:
        logger.warning("Using mock prediction - model not loaded")
        mock_probs = np.random.rand(5)
        mock_probs = mock_probs / mock_probs.sum()
        predicted_class = int(np.argmax(mock_probs))
        confidence = float(mock_probs[predicted_class])
        all_probs = {DR_CLASSES[i]: float(mock_probs[i]) for i in range(5)}
    else:
        try:
            image_bytes = await file.read()
            img_array = preprocess_image(image_bytes)
            predictions = model.predict(img_array)[0]
            predicted_class = int(np.argmax(predictions))
            confidence = float(predictions[predicted_class])
            all_probs = {DR_CLASSES[i]: float(predictions[i]) for i in range(len(predictions))}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    severity_info = get_severity_info(predicted_class)
    
    return ImagePredictionResponse(
        predicted_class=predicted_class,
        class_name=DR_CLASSES[predicted_class],
        confidence=confidence,
        all_probabilities=all_probs,
        severity=severity_info["severity"],
        recommendation=severity_info["recommendation"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "supported_classes": DR_CLASSES
    }

@app.get("/classes")
async def get_classes():
    return {"classes": DR_CLASSES, "description": "Diabetic Retinopathy severity classification"}

@app.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict diabetic retinopathy from retinal image"""
    if model is None:
        logger.warning("Using mock prediction - model not loaded")
        mock_probs = np.random.rand(5)
        mock_probs = mock_probs / mock_probs.sum()
        predicted_class = int(np.argmax(mock_probs))
        confidence = float(mock_probs[predicted_class])
        all_probs = {DR_CLASSES[i]: float(mock_probs[i]) for i in range(5)}
    else:
        try:
            image_bytes = await file.read()
            img_array = preprocess_image(image_bytes)
            predictions = model.predict(img_array)[0]
            predicted_class = int(np.argmax(predictions))
            confidence = float(predictions[predicted_class])
            all_probs = {DR_CLASSES[i]: float(predictions[i]) for i in range(len(predictions))}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    severity_info = get_severity_info(predicted_class)
    
    return ImagePredictionResponse(
        predicted_class=predicted_class,
        class_name=DR_CLASSES[predicted_class],
        confidence=confidence,
        all_probabilities=all_probs,
        severity=severity_info["severity"],
        recommendation=severity_info["recommendation"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)