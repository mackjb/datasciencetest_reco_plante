from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import json
from typing import Dict, Any

app = FastAPI(title="API de Reconnaissance de Plantes et Maladies")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger les modèles
def load_models() -> Dict[str, Any]:
    models = {}
    base_dir = "../results"
    
    for task in ["especes", "maladies"]:
        task_dir = os.path.join(base_dir, f"automl_{task}")
        model_files = [f for f in os.listdir(task_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
        
        if model_files:
            model_path = os.path.join(task_dir, sorted(model_files, reverse=True)[0])
            models[task] = {
                'model': joblib.load(model_path),
                'metadata': {}
            }
            
            # Charger les métadonnées
            summary_files = [f for f in os.listdir(task_dir) if f.startswith('summary_') and f.endswith('.json')]
            if summary_files:
                with open(os.path.join(task_dir, sorted(summary_files, reverse=True)[0]), 'r') as f:
                    models[task]['metadata'] = json.load(f)
    
    return models

# Charger les modèles au démarrage
models = load_models()

@app.get("/")
async def root():
    return {"message": "API de reconnaissance de plantes et maladies - Documentation disponible sur /docs"}

@app.get("/model_info/{task}")
async def get_model_info(task: str):
    if task not in models:
        raise HTTPException(status_code=404, detail=f"Modèle {task} non trouvé")
    
    return {
        "task": task,
        "model_type": str(type(models[task]['model']).__name__),
        "metadata": models[task].get('metadata', {})
    }

# Note: Les endpoints de prédiction nécessiteront l'implémentation de la logique d'extraction de caractéristiques
# qui doit correspondre à celle utilisée pendant l'entraînement.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
