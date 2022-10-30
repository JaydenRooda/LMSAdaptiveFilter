import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

origin = cv2.imread('image.png')
img = cv2.imread('image.png', 0)
cv2.imwrite('image_grayscale.png', img)

example_arr = np.array([
    [0.9361, 1.000,  1.000,  0.8871],
    [1.000,  1.000,  0.9184, 1.000],
    [0.9868, 1.000,  1.000,  0.9591],
    [0.000,  0.8987, 0.9400, 1.000]
])

# img.shape[0] = image height in pixels
# img.shape[1] = image width in pixels
# img.size = total number of pixels in image (= img.shape[0] * img.shape[1])

m = 0  # mean of Gaussian noise
sd = 0.4  # standard deviation of Gaussian noise

# Create noise in the image
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        img[i][j] = img[i][j] + np.random.normal(m, sd)

# Save image with noise
cv2.imwrite('image_noise.png', img)

# Define the window size mxn
M = 3
N = 3

C = np.pad(img, 1, mode='constant')
local_mean = np.empty((img.shape[0], img.shape[1]))
local_variance = np.empty((img.shape[0], img.shape[1]))

def mean(arr):
    s = 0
    for index in range(0, len(arr)):
        s += np.sum(arr[index])
    return s / arr.size

def arr_sum(arr):
    s = 0
    for index in range(0, len(arr)):
        s += np.sum(arr[index])
    return s


for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        window = C[i:i+M, j:j+N]
        local_mean[i][j] = mean(window)
        local_variance[i][j] = mean(np.square(window)) - np.square(mean(window))

noise_variance = arr_sum(local_variance)/img.size

variance = max(noise_variance, arr_sum(local_variance))

img_new = img - (np.multiply(noise_variance/variance, np.subtract(img, local_mean)))

cv2.imwrite('image_restored.png', img_new)
