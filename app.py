from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import os
import sys
import numpy as np
from typing import List, Dict, Any

# Import your diffusion classes
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')
sys.path.append(src_path)
from src.diffusion import DiffusionModel, SpikingDiffusionModel

# Pydantic models for request/response
class GenerateRequest(BaseModel):
    model_name: str
    n_samples: int = 100

class GenerateResponse(BaseModel):
    model_name: str
    n_samples: int
    paths_shape: List[int]
    paths: List[List[float]]  # Return first 10 paths as example

class ModelInfo(BaseModel):
    name: str
    type: str
    sequence_length: int
    description: str

class ModelsResponse(BaseModel):
    available_models: List[str]
    models: Dict[str, ModelInfo]

# Model registry - define your models here
MODEL_REGISTRY = {
    "gbm_low_vol": {
        "path": "models/gbm_low_vol.pt",
        "type": "non_spiking",
        "sequence_length": 64,
        "description": "GBM model with low volatility (sigma=0.1)"
    },
    "gbm_med_vol": {
        "path": "models/gbm_med_vol.pt",
        "type": "non_spiking",
        "sequence_length": 64,
        "description": "GBM model with medium volatility (sigma=0.3)"
    },
    "gbm_high_vol": {
        "path": "models/gbm_high_vol.pt", 
        "type": "non_spiking",
        "sequence_length": 64,
        "description": "GBM model with high volatility (sigma=0.5)"
    }
}

# Global variable to store loaded models
loaded_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load all models
    print("🚀 Loading diffusion models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for model_name, model_info in MODEL_REGISTRY.items():
        try:
            model_path = model_info["path"]
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                continue
            
            # Initialize model architecture
            if model_info["type"] == "spiking":
                model = SpikingDiffusionModel(
                    n_steps=1000,
                    sequence_length=model_info["sequence_length"],
                    device=device
                )
            else:
                model = DiffusionModel(
                    n_steps=1000, 
                    sequence_length=model_info["sequence_length"],
                    device=device
                )
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=device)
            model.model.load_state_dict(checkpoint)
            model.model.to(device)
            model.model.eval()
            
            loaded_models[model_name] = {
                'model': model,
                'info': model_info
            }
            print(f"✅ Loaded: {model_name}")
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)}")
    
    print(f"✅ All models loaded. Total: {len(loaded_models)}")
    yield  # App runs here
    
    # Shutdown: Cleanup
    print("🔚 Shutting down...")
    loaded_models.clear()

app = FastAPI(
    title="Diffusion Model API",
    description="Generate synthetic asset price paths with diffusion models",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "Diffusion Model API - Use /docs for API documentation"}

@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """Get list of all available pre-loaded models"""
    available_models = list(loaded_models.keys())
    
    models_info = {}
    for name, model_data in loaded_models.items():
        models_info[name] = ModelInfo(
            name=name,
            type=model_data['info']['type'],
            sequence_length=model_data['info']['sequence_length'],
            description=model_data['info']['description']
        )
    
    return ModelsResponse(
        available_models=available_models,
        models=models_info
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_paths(request: GenerateRequest):
    """Generate synthetic price paths using specified model"""
    if request.model_name not in loaded_models:
        available_models = list(loaded_models.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{request.model_name}' not found. Available models: {available_models}"
        )
    
    model_data = loaded_models[request.model_name]
    model = model_data['model']
    
    # Generate paths
    with torch.no_grad():
        generated_paths = model.sample(
            n_samples=request.n_samples, 
            batch_size=min(500, request.n_samples),  # Avoid too large batches
            device=next(model.model.parameters()).device
        )
    
    # Convert to numpy and remove channel dimension if present
    paths_np = generated_paths.cpu().numpy()
    if paths_np.ndim == 3:
        paths_np = paths_np.squeeze(1)
    
    # Return first 10 paths as example (avoid huge responses)
    sample_paths = paths_np[:10].tolist()
    
    return GenerateResponse(
        model_name=request.model_name,
        n_samples=request.n_samples,
        paths_shape=list(paths_np.shape),
        paths=sample_paths
    )

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "loaded_models": len(loaded_models),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )