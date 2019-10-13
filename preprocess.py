import numpy as np
from time import time
import argparse
import cv2
from numpy.linalg import solve

import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim

"""
*****************************
Convert a given image to the standard input for inferencing
Author: Derek Zhi
Date: Sep 19, 2019
Version: 0.1
*****************************
"""

def remove_duplicates(lines, image, VIZ=False):
    # filter the redundant
    lines_filt = []
    pos_hori, pos_vert = 0, 0
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if b > 0.5:
            if rho - pos_hori > 10:
                pos_hori = rho
                lines_filt.append([rho, theta, 0])
                if VIZ:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            if rho - pos_vert > 10:
                pos_vert = rho
                lines_filt.append([rho, theta, 1])
                if VIZ:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if VIZ:
        cv2.imshow('image', image)
    if len(lines_filt) != 20:
        raise ValueError("Num of filtered lines should equal 20, but actual num is {0}".format(len(lines_filt)))
    return lines_filt

def get_lines(edges, image, MIN_LINE_LENGTH):
    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, MIN_LINE_LENGTH)
    # Print and draw line on the original image
    lines = np.squeeze(lines, axis=1)
    # sort candidate lines in ascending order
    lines = sorted(lines, key=lambda row: row[0])
    lines = remove_duplicates(lines, image, VIZ=True)
    return lines

def get_points(lines):
    points = []
    for i in range(len(lines)):
        if lines[i][2] == 0:
            for j in range(len(lines)):
                if lines[j][2] == 1:
                    rho1, theta1 = lines[i][:-1]
                    rho2, theta2 = lines[j][:-1]
                    xy = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]],
                                  dtype=np.float32)
                    rho = np.array([rho1, rho2])
                    res = solve(xy, rho)
                    points.append(res)
    print("length of points: ", len(points))
    return points

def save_figs(thresholded, points, VIZ=False):
    for i in range(9):
        for j in range(9):
            y1 = int(points[j + i * 10][1] + 5)
            y2 = int(points[j + i * 10 + 11][1] - 5)
            x1 = int(points[j + i * 10][0] + 5)
            x2 = int(points[j + i * 10 + 11][0] - 5)
            # Saving extracted block for training, uncomment for saving digit blocks
            cv2.imwrite('./data/' + str((i + 1) * (j + 1)) + ".jpg", thresholded[y1: y2,
                                                                     x1: x2])
            if VIZ:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if VIZ:
        plt.show()


def run(img):
    # Loading image contains lines
    img = cv2.imread(img)
    # obtain size of image
    H, W = img.shape[:-1]
    print("Height={0}, Weight={1}".format(H, W))
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    # Blur the image to remove high freq noise
    blurred = cv2.blur(gray, (3, 3))
    cv2.imshow('blurred', blurred)
    # Apply Canny edge detection, return a binary image
    edges = cv2.Canny(blurred, 50, 100, apertureSize=3)
    cv2.imshow('edges', edges)
    MIN_LINE_LENGTH = int(min(H, W) * 0.3) #length of line is at least half of the shorter dim
    lines = get_lines(edges, img, MIN_LINE_LENGTH)
    points = get_points(lines)
    # Thresholding image
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 1)
    save_figs(thresholded, points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    args = parser.parse_args()
    run(args.image)