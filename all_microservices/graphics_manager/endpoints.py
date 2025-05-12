from fastapi import FastAPI, Form
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import requests
from urllib.parse import quote
import io
import graphics_manager_script as GMS
from PIL import Image

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
    return Response("This is a root of graphics manager", status_code=200)

@app.get("/graphics")
def graphics():
    try:
        history_response = requests.get("http://neural_net:5001/get_history", timeout=10)
        history_response.raise_for_status()
        history = history_response.json()

        if not history or not any(history.values()):
            return Response("No training data available", status_code=204)

        buf = GMS.get_graphic(history)

        image_bytes = buf.getvalue()
        buf.close()
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png", status_code=200)
    except requests.exceptions.HTTPError as e:
        return Response(f"Failed to connect: {e}", status_code=500)
    except Exception as e:
        return Response(f"Unexpected error has occured: {e}", status_code=500)