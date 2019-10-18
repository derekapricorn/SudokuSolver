"""
Validate that the trained model is able to perform at good accuracy
which is around above 98%
"""

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import sys
from sklearn.metrics import accuracy_score
from CNN import CNN

model = CNN()
model.load_state_dict(torch.load(sys.argv[1]))
model.eval()
transform = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor(), transforms.Normalize((0.5,) , (0.5,)),])
testset = datasets.MNIST('./data/test', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
testiter = iter(testloader)
for _ in range(4):
    images, labels = testiter.next()
    n_samples = labels.shape[0]
    preds = []
    with torch.no_grad():
        logps = model(images)
    ps = torch.exp(logps)
    preds = np.argmax(ps.numpy(), axis=1)
        # print("{0} vs {1}".format(pred, labels[i]))
    print("accuracy score = ", accuracy_score(preds, labels))