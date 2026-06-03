
"""
FastAPI Backend for Image Caption Generator
"""

import os
import shutil
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from PIL import Image
import uvicorn
import requests

# Load environment variables
load_dotenv()

# ============================
# CONFIGURATION
# ============================
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Model Configuration
MODEL_KEY = os.getenv("MODEL_KEY")

# ============================
# FASTAPI APP
# ============================
app = FastAPI(
    title="Image Caption Generator API",
    description="Upload an image and get an AI-generated caption using our trained vision-language model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CaptionResponse(BaseModel):
    success: bool
    caption: str
    filename: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# ============================
# HELPERS
# ============================
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption(image_path: str) -> str:
    base64_image = encode_image(image_path)
    
    headers = {
        "Authorization": f"Bearer {MODEL_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one clear, concise sentence."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    caption = result["choices"][0]["message"]["content"].strip()
    return caption

def clean_caption(raw: str) -> str:
    caption = raw.strip()
    if caption:
        caption = caption[0].upper() + caption[1:]
        if not caption.endswith("."):
            caption += "."
    return caption

# ============================
# ENDPOINTS
# ============================
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="running", model_loaded=True)

@app.post("/predict", response_model=CaptionResponse)
async def predict(file: UploadFile = File(...)):
    allowed = {"jpg", "jpeg", "png", "webp", "bmp"}
    ext = file.filename.split(".")[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid file. Allowed: {', '.join(allowed)}")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    try:
        img = Image.open(file_path)
        img.verify()
    except Exception:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        raw = generate_caption(file_path)
        clean = clean_caption(raw)
        return CaptionResponse(success=True, caption=clean, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
