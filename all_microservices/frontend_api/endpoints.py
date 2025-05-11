from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
import numpy as np
import io

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    return Response("This is a root of frontend API microservice", status_code=200)

@app.get("/mainpage", response_class=HTMLResponse)
def mainpage(request: Request):
    try:
        history_response = requests.get(f"http://neural_net:5001/get_history")
        history_response.raise_for_status()
        history = history_response.json()
        val_loss = history.get("val_loss", [])
        loss = history.get("loss", [])
        accuracy = history.get("accuracy", [])
        val_accuracy = history.get("val_accuracy", [])
        print(history)
        history_ = [[i+1, loss[i], val_loss[i], accuracy[i], val_accuracy[i]]for i in range(0, len(loss))]
        return templates.TemplateResponse("mainpage.html", {"request": request, "history": history_})
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)
