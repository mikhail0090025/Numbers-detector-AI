from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import neural_net_script as NNS
import numpy as np
import requests
from urllib.parse import quote
import json
import io

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return Response("This is a root of neural net manager", status_code=200)

@app.post("/go_epochs")
async def go_epochs(epochs_count: int = Form(default=10)):
    try:
        print(f"Received epochs_count: {epochs_count}")
        NNS.go_epochs(epochs_count)
        return JSONResponse({"message": f"{epochs_count} Epochs are passed"}, status_code=200)
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse({"error": f"Internal server error: {str(e)}"}, status_code=500)

@app.get("/get_history")
def get_history():
    try:
        return JSONResponse({
            "val_loss": NNS.all_val_losses,
            "loss": NNS.all_losses,
            "accuracy": NNS.all_accuracies,
            "val_accuracy": NNS.all_val_accuracies,
        }, status_code=200)
    except Exception as e:
        return JSONResponse({"error": f"Internal server error: {str(e)}"}, status_code=500)

@app.get("/current_lr")
def get_lr():
    try:
        current_lr = 0
        try:
            current_lr = NNS.model.optimizer.learning_rate.numpy()
        except AttributeError as e:
            current_lr = NNS.start_lr
        return JSONResponse({"current_lr": str(current_lr)}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/predict")
def predict(url: str):
    try:
        image_response = requests.get(f"http://images_manager:5000/image_to_inputs?url={quote(url, safe='')}")
        image_response.raise_for_status()
        image = json.loads(image_response.text)
        image = np.array(image['image'])
        prediction = NNS.model.predict(np.array([image]), verbose=1)
        response = {
            'prediction': prediction.tolist()[0],
            'predicted_number': prediction.tolist()[0].index(max(prediction.tolist()[0])),
        }
        return JSONResponse(json.dumps(response), status_code=200)
    except Exception as e:
        return JSONResponse({'error': e}, status_code=500)