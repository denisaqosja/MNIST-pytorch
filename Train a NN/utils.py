"""
Util methods

"""

import os
import json

import torch

from model import Net


def create_directory(path, name=None):
    """
    Creating a new directory if it does not already exist

    Args:
    -----
    path: string
        path/directory to be created
    name: string/None
        if not None, directory with name 'name' is created in path
    """

    if(name is not None):
        path = os.path.joint(path, name)
    # creating directory only if necesary
    if(not os.path.exists(path)):
        os.makedirs(path)

    return


def export_model():
    """
    Creating the model and saving the architecture in a txt
    """

    return

def save_model(model, path, name):
    """
    Saving the model parameters

    Args:
    -----
    model: torch Module
        model whose parameters we want to save
    path: string
        path to the directory where model will be saved
    name: string
        name of the file containing the parameters
    """

    # enforcing correct values
    assert name[-4:] == ".pth", f"ERROR! Parameters file {name} must end with '.pth'"
    assert os.path.exists(path), f"ERROR! Path {path} does not exist..."

    savepath = os.path.join(path, name)
    torch.save(model.state_dict(), savepath)

    return


def load_model(model, path, name):
    """
    Loading a model from a pre-saved state dictionary

    Args:
    -----
    model: torch Module
        model skeleton initialized with random weights
    path: string
        path to the directory where model is stored
    name: string
        name of the file containing the parameters

    Returns:
    --------
    model: torch Module
        input model but with the parameters loaded from the file
    """

    # enforcing correct file name
    assert name[-4:] == ".pth", f"ERROR! Parameters file {name} must end with '.pth'"

    # making sure file contaning pretrained parameters exists
    fpath = os.path.join(path, name)
    assert os.path.exists(fpath), f"ERROR! File {fpath} containing pretrained parameters "\
                                   "does not exist..."

    model.load_state_dict(torch.load(fpath))

    return model


def save_stats(train_loss, valid_loss, train_acc, valid_acc):
    """
    Saving training statistics in a json file for logging purposes
    """

    savepath = os.path.join(os.getcwd(), "training_logs.json")
    logs = {
        "loss": {
            "train": train_loss,
            "valid": valid_loss,
        },
        "accuracy": {
            "train": train_acc,
            "valid": valid_acc,
        }
    }
    with open(savepath, "w") as f:
        json.dump(logs, f)
    return

#
