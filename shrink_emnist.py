# Import required packages
from scipy.io import loadmat,savemat
import numpy as np
import os

# Import file and initialize parameters
folder = input("Please provide the folder to the emnist-balanced.mat file")
data = loadmat(os.path.join(folder,'emnist-balanced.mat'))
shrinksize = 2000

# Reduce dataset size
data.get("dataset")[0][0][0][0][0][0] = data.get("dataset")[0][0][0][0][0][0][:shrinksize]
data.get("dataset")[0][0][0][0][0][1] = data.get("dataset")[0][0][0][0][0][1][:shrinksize]
data.get("dataset")[0][0][0][0][0][2] = data.get("dataset")[0][0][0][0][0][2][:shrinksize]
data.get("dataset")[0][0][1][0][0][0] = data.get("dataset")[0][0][1][0][0][0][:shrinksize]
data.get("dataset")[0][0][1][0][0][1] = data.get("dataset")[0][0][1][0][0][1][:shrinksize]
data.get("dataset")[0][0][1][0][0][2] = data.get("dataset")[0][0][1][0][0][2][:shrinksize]

# Save data to file
savemat(os.path.join(folder,'emnist-balanced-small.mat'),data)