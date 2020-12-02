"""
Exporting the model architecture into a txt file
"""

import os

from model import Net

def export_model():
    """
    Creating the model and saving the architecture in a txt
    """

    model = Net()
    network_file = "network_architecture.txt"
    with open(network_file, "w") as file:
        for layer in model.layers:
            file.write(str(layer))
            file.write("\n")
    return

if __name__ == "__main__":
    os.system("clear")
    export_model()
