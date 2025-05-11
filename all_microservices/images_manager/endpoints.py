from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import os
import requests
from PIL import Image, UnidentifiedImageError

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return Response("This is a root of images manager", status_code=200)

import images_manager_script as IMS
IMS.get_images()

@app.get("/is_ready")
def is_ready():
    return Response(str(IMS.images_are_loaded), status_code=200)

@app.get("/images")
def get_images(page: int = 0, per_page: int = 100):
    if not IMS.images_are_loaded:
        return Response("Images are not ready yet", status_code=500)
    start = page * per_page
    end = start + per_page
    if start >= len(IMS.images):
        return {"error": "Page out of range"}
    if end > len(IMS.images):
        end = len(IMS.images)
    buffer = io.BytesIO()
    np.save(buffer, IMS.images[start:end])
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename=images_page_{page}.npy"})

@app.get("/outputs")
def get_outputs(page: int = 0, per_page: int = 100):
    if not IMS.images_are_loaded:
        return Response("Images are not ready yet", status_code=500)
    start = page * per_page
    end = start + per_page
    if start >= len(IMS.outputs):
        return {"error": "Page out of range"}
    if end > len(IMS.outputs):
        end = len(IMS.outputs)
    # Сохраняем в буфер
    buffer = io.BytesIO()
    np.save(buffer, IMS.outputs[start:end])
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename=outputs_page_{page}.npy"})

@app.get("/image_to_inputs")
async def image_to_inputs(url: str):
    if not url:
        return JSONResponse({"error": "URL cannot be empty"}, status_code=400)

    try:
        image_response = requests.get(url, timeout=10)
        image_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Failed to fetch image from URL: {e}"}, status_code=400)

    try:
        img = Image.open(io.BytesIO(image_response.content))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img_resized = img.resize((100, 100), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        if img_array.shape != (100, 100, 3):
            return JSONResponse({"error": f"Expected image shape (100, 100, 3), got {img_array.shape}"}, status_code=400)
        
        img_array_list = img_array.tolist()
        
        return JSONResponse({"image": img_array_list}, status_code=200)
    
    except UnidentifiedImageError:
        return JSONResponse({"error": f"Error loading {url}: not a valid image"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Unexpected error: {str(e)}"}, status_code=500)
