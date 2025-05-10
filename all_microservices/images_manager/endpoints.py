from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
import images_manager_script as IMS
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io

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

@app.get("/images")
def get_images(page: int = 0, per_page: int = 100):
    while not IMS.images_are_loaded:
        pass
    start = page * per_page
    end = start + per_page
    if start >= len(IMS.images):
        return {"error": "Page out of range"}
    if end > len(IMS.images):
        end = len(IMS.images)
    # Сохраняем в буфер
    buffer = io.BytesIO()
    np.save(buffer, IMS.images[start:end])
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename=images_page_{page}.npy"})

@app.get("/outputs")
def get_outputs(page: int = 0, per_page: int = 100):
    while not IMS.images_are_loaded:
        pass
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
