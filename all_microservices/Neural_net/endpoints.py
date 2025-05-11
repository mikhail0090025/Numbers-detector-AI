from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import neural_net_script as NNS
import numpy as np

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