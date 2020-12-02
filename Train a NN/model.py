import torch.nn as nn
import torch.nn.functional as F
from data import load_train_dataset, load_test_dataset
import torch
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        layer1 = self._make_layer(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        layers.append(layer1)
        layer2 = self._make_layer(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        layers.append(layer2)
        layer3 = self._make_layer(in_channels=64, out_channels=128, kernel_size=5, stride=1)
        layers.append(layer3)

        flatten = nn.Flatten()
        layers.append(flatten)
        fc1 = nn.Linear(in_features=128*4*4, out_features=128)
        relu = nn.ReLU(True)
        layers.append(fc1)
        layers.append(relu)
        fc2 = nn.Linear(in_features=128, out_features=10)
        # softmax = nn.Softmax(dim=1)
        layers.append(fc2)
        # layers.append(softmax)
        self.layers = nn.Sequential(*layers)
        return


    def _output_size(self, input_size, kernel_size, stride, padding_size):
        return np.floor((input_size-kernel_size +2*padding_size)/stride) +1

    def _padding_size(self, input_size, kernel_size, stride):
        """
        return the padding size for the images;
        desired: same input and output size
        P = ((S-1)*W-S+F)/2
        """
        return ((stride - 1)*input_size - stride + kernel_size)//2

    def _make_layer(self, in_channels, out_channels, kernel_size, stride):
        padd_size = self._padding_size(in_channels, kernel_size, stride)
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padd_size)
        relu = nn.ReLU(True)
        pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        layer = nn.Sequential(conv, relu, pooling)
        return layer


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
