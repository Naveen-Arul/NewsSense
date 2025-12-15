# ======================================================
# FASTAPI ML INFERENCE API
# ------------------------------------------------------
# Serves trained news classification models via REST API
# Loads all models once at startup for fast inference
# ======================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import joblib
import os
import re
from typing import List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

# Load environment variables from .env file
load_dotenv()

# ======================================================
# GLOBAL VARIABLES FOR MODELS AND PREPROCESSORS
# ======================================================
models = {}
vectorizer = None
target_names = None
stop_words = None
lemmatizer = None

# Model performance metrics (from training results)
MODEL_METRICS = {
    "Naive Bayes": {"accuracy": 0.7135, "f1_score": 0.6853},
    "Logistic Regression": {"accuracy": 0.7340, "f1_score": 0.7112},
    "Support Vector Machine": {"accuracy": 0.7489, "f1_score": 0.7289},
    "Random Forest": {"accuracy": 0.6623, "f1_score": 0.6234}
}

# Best model based on F1-score
BEST_MODEL = "Support Vector Machine"

# ======================================================
# REQUEST/RESPONSE MODELS
# ======================================================
class PredictionRequest(BaseModel):
    text: str

class ModelPrediction(BaseModel):
    model: str
    prediction: str
    prediction_confidence: float

class PredictionResponse(BaseModel):
    input_text: str
    cleaned_text: str
    results: List[ModelPrediction]
    best_model: str

# ======================================================
# LIFESPAN EVENT: LOAD ALL MODELS ONCE
# ======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all trained models and TF-IDF vectorizer at server startup.
    This ensures models are loaded only once for efficiency.
    """
    global models, vectorizer, target_names, stop_words, lemmatizer
    
    models_dir = "models"
    
    print("\n" + "="*60)
    print("LOADING MODELS AT STARTUP...")
    print("="*60)
    
    try:
        # Load Naive Bayes
        models["Naive Bayes"] = joblib.load(
            os.path.join(models_dir, "naive_bayes_model.pkl")
        )
        print("✓ Loaded: Naive Bayes")
        
        # Load Logistic Regression
        models["Logistic Regression"] = joblib.load(
            os.path.join(models_dir, "logistic_regression_model.pkl")
        )
        print("✓ Loaded: Logistic Regression")
        
        # Load Support Vector Machine
        models["Support Vector Machine"] = joblib.load(
            os.path.join(models_dir, "svm_model.pkl")
        )
        print("✓ Loaded: Support Vector Machine")
        
        # Load Random Forest
        models["Random Forest"] = joblib.load(
            os.path.join(models_dir, "random_forest_model.pkl")
        )
        print("✓ Loaded: Random Forest")
        
        # Load TF-IDF Vectorizer
        vectorizer = joblib.load(
            os.path.join(models_dir, "tfidf_vectorizer.pkl")
        )
        print("✓ Loaded: TF-IDF Vectorizer")
        
        # Initialize NLP tools
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        print("✓ Initialized: NLTK stopwords and lemmatizer")
        
        # 20 Newsgroups target names
        target_names = [
            'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
            'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
            'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
            'sci.space', 'soc.religion.christian', 'talk.politics.guns',
            'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
        ]
        
        print("="*60)
        print("ALL MODELS LOADED SUCCESSFULLY!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise
    
    yield
    
    # Cleanup (runs on shutdown)
    print("\n" + "="*60)
    print("SHUTTING DOWN SERVER...")
    print("="*60)

# ======================================================
# INITIALIZE FASTAPI APP
# ======================================================
app = FastAPI(
    title="News Classification ML API",
    description="Multi-model news topic classification using NLP and Machine Learning",
    version="1.0.0",
    lifespan=lifespan
)

# ======================================================
# CORS CONFIGURATION - Allow frontend to access API
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8080").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# NLP PREPROCESSING FUNCTION
# ------------------------------------------------------
# EXACT SAME LOGIC AS TRAINING SCRIPT
# Ensures consistency between training and inference
# ======================================================
def preprocess_user_input(text: str) -> str:
    """
    Apply complete NLP preprocessing pipeline to user input.
    
    Steps:
    1. Lowercase
    2. URL removal
    3. Emoji removal (complete Unicode range)
    4. Repeated character normalization
    5. Special character & number removal
    6. Tokenization
    7. Stopword removal
    8. Lemmatization
    
    Args:
        text: Raw user input text
        
    Returns:
        Cleaned and preprocessed text ready for TF-IDF vectorization
    """
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. URL removal
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 3. Emoji removal (complete Unicode coverage)
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"  # supplemental symbols (includes 🤯)
        "\U0001FA00-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    
    # 4. Normalize repeated characters (e.g., "cooool" -> "cool")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 5. Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 6. Tokenization
    tokens = word_tokenize(text)
    
    # 7. Stopword removal (keep only words > 2 characters)
    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    
    # 8. Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]
    
    # Return final cleaned text
    return " ".join(tokens)

# ======================================================
# ROOT ENDPOINT
# ======================================================
@app.get("/")
async def root():
    """
    API health check and information endpoint.
    """
    return {
        "message": "News Classification ML API",
        "status": "running",
        "models_loaded": len(models),
        "available_models": list(models.keys()),
        "endpoint": "/predict (POST)"
    }

# ======================================================
# PREDICTION ENDPOINT
# ======================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict news topic from input text using all trained models.
    
    Args:
        request: JSON body containing 'text' field
        
    Returns:
        JSON response with predictions from all models and performance metrics
    """
    
    try:
        # Validate input
        if not request.text or request.text.strip() == "":
            raise HTTPException(
                status_code=400, 
                detail="Input text cannot be empty"
            )
        
        # Apply NLP preprocessing
        cleaned_text = preprocess_user_input(request.text)
        
        # Check if cleaned text is empty after preprocessing
        if not cleaned_text or cleaned_text.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="Text contains no valid content after preprocessing"
            )
        
        # Convert to TF-IDF features
        text_vector = vectorizer.transform([cleaned_text])
        
        # Get predictions from all models
        results = []
        
        for model_name, model in models.items():
            # Predict
            prediction_index = model.predict(text_vector)[0]
            prediction_label = target_names[prediction_index]
            
            # Get prediction probability/confidence
            try:
                # For models that support predict_proba
                probabilities = model.predict_proba(text_vector)[0]
                prediction_confidence = float(probabilities[prediction_index])
            except AttributeError:
                # For models without predict_proba (like SVM with default kernel)
                # Use decision function and normalize
                decision = model.decision_function(text_vector)[0]
                # Normalize to 0-1 range using softmax-like approach
                exp_decision = np.exp(decision - np.max(decision))
                prediction_confidence = float(exp_decision[prediction_index] / exp_decision.sum())
            
            # Get metrics from training
            metrics = MODEL_METRICS[model_name]
            
            # Add to results
            results.append(ModelPrediction(
                model=model_name,
                prediction=prediction_label,
                prediction_confidence=round(prediction_confidence, 4)
            ))
        
        # Build response
        response = PredictionResponse(
            input_text=request.text,
            cleaned_text=cleaned_text,
            results=results,
            best_model=BEST_MODEL
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# ======================================================
# MODEL INFO ENDPOINT
# ======================================================
@app.get("/models")
async def get_models_info():
    """
    Get information about all loaded models and their performance.
    """
    return {
        "total_models": len(models),
        "models": [
            {
                "name": name,
                "accuracy": MODEL_METRICS[name]["accuracy"],
                "f1_score": MODEL_METRICS[name]["f1_score"]
            }
            for name in models.keys()
        ],
        "best_model": BEST_MODEL,
        "categories": target_names
    }

# ======================================================
# RUN SERVER
# ======================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
