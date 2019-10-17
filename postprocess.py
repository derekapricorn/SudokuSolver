"""
Given a folder of images, return a matrix of numbers
Author: derekz
Date: 10-17-2019
"""
import os, glob
import numpy as np
from predict import predict

def imgs_to_mat(data_folder, format='.jpg'):
    res = np.zeros((9,9))
    files = glob.glob(os.path.join(data_folder, '*' + format))
    if not files:
        print("can't find any image files")
        exit(-1)
    for f in files:
        filename = os.path.basename(f).rstrip(format)
        row, col = map(int, filename.split('_'))
        val = predict(f)
        res[row, col] = val
    print(res)

imgs_to_mat('data/temp')

