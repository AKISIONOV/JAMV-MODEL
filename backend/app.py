from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from tensorflow import keras
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JAMV MODEL API", description="ML Model Prediction API")

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

try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: list
    confidence: list
    predicted_class: str

@app.get("/")
async def root():
    return {"message": "JAMV MODEL API", "status": "running", "model_loaded": model is not None}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        features = np.array(request.features)
        
        # Reshape if needed (batch_size, features)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        predictions = model.predict(features)
        
        # Get predicted class
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0].tolist()
        
        return PredictionResponse(
            prediction=predictions[0].tolist(),
            confidence=confidence,
            predicted_class=str(predicted_class)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert all features to numpy array
        features_list = [np.array(req.features) for req in requests]
        features = np.array(features_list)
        
        # Make predictions
        predictions = model.predict(features)
        
        results = []
        for pred in predictions:
            predicted_class = np.argmax(pred)
            results.append({
                "prediction": pred.tolist(),
                "confidence": pred.tolist(),
                "predicted_class": str(predicted_class)
            })
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)