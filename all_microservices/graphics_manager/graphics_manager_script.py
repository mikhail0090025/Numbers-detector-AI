import matplotlib.pyplot as plt
import numpy as np
import io

image_path = "training_metrics.png"

def get_graphic(history):
    history_keys = history.keys()
    
    plt.figure(figsize=(10, 6))
    for key in history_keys:
        parameter = history[key]
        epochs = range(1, len(parameter) + 1)
        plt.plot(epochs, parameter, label=key, marker='o')
    
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf