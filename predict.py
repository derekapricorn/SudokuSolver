import imutils
import sys
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import cv2
from torchvision.transforms import ToTensor, Normalize


def toImage(net_output):
    img = net_output.data.squeeze().permute(1, 2, 0).numpy()  # [1,c,h,w]->[h,w,c]
    img = (img*255.0).clip(0, 255)
    img = numpy.uint8(img)
    img = Image.fromarray(img, mode='RGB')
    return img

def validate_input(img_in):
    img = cv2.imread(img_in)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = imutils.resize(gray, width=28)
    plt.imshow(imutils.opencv2matplotlib(resized))
    plt.show()

img_in = sys.argv[2]
img = cv2.imread(img_in)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28, 28))
img = ToTensor()(resized)
img = Normalize((0.5,) , (0.5,))(img)
net_input = Variable(img).view(1, 784)


input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(*hidden_sizes),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1)
                      )

model.load_state_dict(torch.load(sys.argv[1]))
model.eval()
with torch.no_grad():
    logps = model(net_input )
ps = torch.exp(logps)
pred = np.argmax(ps.numpy()[0])
print("predicting ", pred)


