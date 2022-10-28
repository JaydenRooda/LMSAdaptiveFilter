import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

img = cv2.imread('image.png', 0)
cv2.imwrite('image_grayscale.png', img)
sz = img.shape[0] * img.shape[1]

m = 0           # mean of Gaussian noise
sd = 0.5      # standard deviation of Gaussian noise

i = 0
while i < img.shape[0]:
    j = 0
    while j < img.shape[1]:
        img[i][j] = img[i][j] + np.random.normal(m, sd)
        j += 1
    i += 1

cv2.imwrite('image_noisey.png', img)


# Define the window size mxn
M = 5
N = 5

# Pad the array with zeros on all sides
C = np.pad(img, )

print(img[0])
print(C[0])
