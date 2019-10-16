import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) , (0.5,)),])
# trainset = datasets.MNIST('./data/train', download=True, train=True, transform=transform)
# testset = datasets.MNIST('./data/test', download=True, train=False, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# input_size = 784
# hidden_sizes = [128, 64]
# output_size = 10
#
# model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
#                       nn.ReLU(),
#                       nn.Linear(*hidden_sizes),
#                       nn.ReLU(),
#                       nn.Linear(hidden_sizes[1], output_size),
#                       nn.LogSoftmax(dim=1)
#                       )
# print(model)
#
# criterion = nn.NLLLoss()
# images = images.view(images.shape[0],-1)
# logps = model(images)
# loss = criterion(logps, labels)
# print('Before backward pass: \n', model[0].weight.grad)
# loss.backward()
# print('After backward pass: \n', model[0].weight.grad)

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for imgs, lbls in trainloader:
        imgs = imgs.view(imgs.shape[0], -1)
        optimizer.zero_grad() # clear the gradients
        output = model(imgs)
        loss = criterion(output, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("epoch {0} - Training loss : {1}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)

images, labels = next(iter(testloader))
img = images[0].view(1, 784)
with torch.no_grad():
    model.eval()
    logps = model(img)

ps = torch.exp(logps)

probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
#view_classify(img.view(1, 28, 28), ps)

torch.save(model.state_dict(), './models/my_mnist_model.pth')
