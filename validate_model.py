import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import sys

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(*hidden_sizes),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1)
                      )w

model.load_state_dict(torch.load(sys.argv[1]))
model.eval()
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) , (0.5,)),])
testset = datasets.MNIST('./data/test', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
testiter = iter(testloader)
images, labels = testiter.next()
for i in range(64):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)
    ps = torch.exp(logps)
    pred = np.argmax(ps.numpy()[0])
    print("{0} vs {1}".format(pred, labels[i]))