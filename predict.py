"""
Given a model and an image, predict what digit the image contains
"""
import imutils
import sys
import numpy as np
import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import cv2
from torchvision.transforms import ToTensor, Normalize
from CNN import CNN


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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file")
    parser.add_argument("--model")
    args = parser.parse_args()
    print("load input image")
    image = cv2.imread(args.image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    net_input = Normalize((0.5,) , (0.5,))(ToTensor()(resize))

    model = CNN()
    model.load_state_dict(torch.load(args.model))
    model.eval()
    with torch.no_grad():
        logps = model(net_input)
        ps = torch.exp(logps)
        pred = np.argmax(ps.numpy()[0])
        print("predicting ", pred)


