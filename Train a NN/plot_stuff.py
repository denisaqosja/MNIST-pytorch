import os
import json

import numpy as np
from matplotlib import pyplot as plt

def main():
    """
    """

    fname = os.path.join(os.getcwd(), "training_logs.json")
    savepath = os.path.join(os.getcwd(), "train_landscape.png")
    with open(fname) as f:
        logs = json.load(f)

    train_loss = logs["loss"]["train"][1:]
    valid_loss = logs["loss"]["valid"][1:]
    train_acc = logs["accuracy"]["train"][1:]
    valid_acc = logs["accuracy"]["valid"][1:]
    epochs = np.arange(1, len(train_loss)+1)

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(16,7)
    ax[0].plot(epochs, train_loss, linewidth=3, label="train loss")
    ax[0].plot(epochs, valid_loss, linewidth=3, label="valid loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid()
    ax[0].legend(loc="best")

    ax[1].plot(epochs, train_acc, linewidth=3, label="train acc")
    ax[1].plot(epochs, valid_acc, linewidth=3, label="valid acc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid()
    ax[1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(savepath)

    return

if __name__ == "__main__":
    os.system("clear")
    main()
