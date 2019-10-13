import imutils
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy.linalg import solve
import sys

def plot(img):
    plt.imshow(imutils.opencv2matplotlib(img))
    plt.show()


# Loading image contains lines
img = cv2.imread(sys.argv[1])
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.blur(gray, (3, 3))
# Apply Canny edge detection, return will be a binary image
edges = cv2.Canny(blurred, 50, 100, apertureSize=3)
# Apply Hough Line Transform, minimum lenght of line is 200 pixels
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
# Print and draw line on the original image
lines = np.squeeze(lines, axis=1)
lines = sorted(lines, key=lambda row: row[0])
new_lines = []
points = []
pos_hori, pos_vert = 0, 0
for line in lines:
    rho, theta = line
    print(rho, theta)
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
            new_lines.append([rho, theta, 0])
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        if rho - pos_vert > 10:
            pos_vert = rho
            new_lines.append([rho, theta, 1])
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
print("length of new_lines is :",len(new_lines))
for i in range(len(new_lines)):
    if new_lines[i][2] == 0:
        for j in range(len(new_lines)):
            if new_lines[j][2] == 1:
                rho1, theta1 = new_lines[i][:-1]
                rho2, theta2 = new_lines[j][:-1]
                xy = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]], dtype=np.float32)
                rho = np.array([rho1, rho2])
                res = solve(xy, rho)
                points.append(res)
print("length of points: ", len(points))
sudoku1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 101, 1)
for i in range(9):
    for j in range(9):
        y1 = int(points[j + i * 10][1] + 5)
        y2 = int(points[j + i * 10 + 11][1] - 5)
        x1 = int(points[j + i * 10][0] + 5)
        x2 = int(points[j + i * 10 + 11][0] - 5)
        # Saving extracted block for training, uncomment for saving digit blocks
        cv2.imwrite('./data/' + str((i+1)*(j+1))+".jpg", sudoku1[y1: y2,
                                                   x1: x2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
plot(img)
# Show the result
# plot(img)
