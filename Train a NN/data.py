import torch
import torchvision as tv

#MNIST datasets
data_transforms = tv.transforms.Compose([tv.transforms.Pad(2), tv.transforms.ToTensor()])

def load_train_dataset():
    train = tv.datasets.MNIST("", train=True, download=True,
                                transform = data_transforms)
    train_dataset = torch.utils.data.DataLoader(dataset=train, batch_size=128, shuffle=True)
    return train_dataset


def load_test_dataset():
    test = tv.datasets.MNIST("", train=False, download=True,
                                transform = data_transforms)
    test_dataset = torch.utils.data.DataLoader(dataset=test, batch_size=128, shuffle=True)
    return test_dataset

