import matplotlib.pyplot as plt
import numpy as np

def get_graphic(history):
    history = history.history
    historyKeys = history.keys()
    for key in historyKeys:
        parameter = history[key]
        plt.plot(parameter, [i for i in range(len(parameter))], label = key)
    plt.legend()
    plt.grid(True)