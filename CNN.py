"""
Train a CNN classifier for digit recognition
Author: derekz
Date: 10/15/2019
"""
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from datetime import date
today = date.today()


LR=0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), ]
    )
    trainset = datasets.MNIST('./data/train', download=True, train=True, transform=transform)
    testset = datasets.MNIST('./data/test', download=True, train=False, transform=transform)
    print("training size = {0}, testing size = {1}".format(trainset.train_data.size(), testset.test_data.size()))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return trainloader, testloader

class CNN(nn.Module):
    def __init__(self, ks=5, s=1, p=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=ks,  # filter size
                stride=s,  # filter movement/step
                padding=p,
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, ks, s, p),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


def train(model, train_loader, test_loader, optimizer, loss_fnc, EPOCH=100):
    MIN_LOSS = float('inf')
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            x.to(device)
            y.to(device)
            output = model(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            images, labels = next(iter(test_loader))
            images.to(device)
            labels.to(device)
            output_test = model(images)
            loss_test = loss_func(output_test, labels)
            if loss_test < MIN_LOSS * 0.5:
                MIN_LOSS = loss_test
                torch.save(model.state_dict(), "./models/{0}.pth".format(today.strftime("%m-%d-%Y")))
            print("training loss = {0}; test loss = {1}".format(loss.data.numpy(), loss_test.data.numpy()))

if __name__ == '__main__':
    cnn = CNN()
    cnn.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    trainloader, testloader = load_data()
    train(cnn, trainloader, testloader, optimizer, loss_func)