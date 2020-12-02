"""
Training a CNN to be trained on mnist
"""
import os
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.nn import Softmax
import torch.optim as optim

from model import Net
from utils import save_stats
from data import load_train_dataset, load_test_dataset

class Trainer:
    """
    Loads dataset, loads model, trains model
    """

    def __init__(self):
        """
        Initializer of the trainer object
        """

        self.train_loss = 1e6
        self.valid_loss = 1e6
        self.train_acc = 0
        self.valid_acc = 0
        return

    def load_dataset(self):
        """
        Loading train, validation an test datasets
        """
        self.train_dataloader = load_train_dataset()
        self.test_dataloader = load_test_dataset()


    def setup_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Net()
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4, betas=(0.95, 0.999))
        self.model = self.model.to(self.device)
        self.softmax = Softmax(dim=1)
        return


    def train_model(self):
        """
        """
        self.train_loss, self.train_acc = [], []
        self.valid_loss, self.valid_acc = [], []
        for epoch in range(10):
            valid = self.valid_epoch()
            print(f"Validation Accuracy: {self.valid_acc[-1]}")
            print(f"Validation Loss: {self.valid_loss[-1]}")
            train = self.train_epoch()
            print(f"Training Accuracy: {self.train_acc[-1]}")
            print(f"Training Loss: {self.train_loss[-1]}")
            save_stats(train_loss=self.train_loss, valid_loss=self.valid_loss,
                       train_acc=self.train_acc, valid_acc=self.valid_acc)
        valid = self.valid_epoch()
        print(f"Final Accuracy: {self.valid_acc[-1]}")
        print(f"Final Loss: {self.valid_loss[-1]}")

        return

    def train_epoch(self):
        """
        """
        train_losses = []
        correct_class = 0
        self.model.train()

        for batch_id, (data, label) in enumerate(tqdm(self.train_dataloader)):
            self.optimizer.zero_grad()
            data, label = data.to(self.device), label.to(self.device)
            output = self.model(data)
            loss = self.loss(output, label)
            loss.backward()
            train_losses.append(loss.item())

            self.optimizer.step()

            # prediction = torch.argmax(output, axis=1)
            prediction = torch.argmax(self.softmax(output), axis=1)
            correct_class += prediction.eq(label).sum().item()

        total_length = len(self.train_dataloader.dataset)
        self.train_acc.append(100 * correct_class / total_length)
        self.train_loss.append(np.mean(train_losses))

        return


    def valid_epoch(self):
        self.model.eval()

        correct_class = 0
        valid_losses = []
        with torch.no_grad():
            for batch_id, (data, label) in enumerate(self.test_dataloader):
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                loss = self.loss(output, label)
                valid_losses.append(loss.item())
                prediction = torch.argmax(self.softmax(output), axis=1)
                # prediction = torch.argmax(output, axis =1)
                correct_class += prediction.eq(label).sum().item()

            total_length = len(self.test_dataloader.dataset)
            self.valid_acc.append(100*correct_class / total_length)
            self.valid_loss.append(np.mean(valid_losses))

        return



if __name__ == "__main__":
    os.system("clear")
    train = Trainer()
    train.load_dataset()
    train.setup_model()
    train.train_model()
