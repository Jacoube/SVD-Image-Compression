from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
plt.rcParams['figure.figsize'] = [16, 8]

A = imread('comet_crop.jpg')
X = np.mean(A, -1)  # Convert RGB to grayscale

# 0 - red; 1 - green; 2 - blue
# Extract only the color channel
red_channel = A[:, :, 0]
green_channel = A[:, :, 1]
blue_channel = A[:, :, 2]

# Create an all-zero image with the same shape as the original image
red_image = np.zeros_like(A)
green_image = np.zeros_like(A)
blue_image = np.zeros_like(A)

# Set the red channel of the new image to the extracted red channel values
red_image[:, :, 0] = red_channel
green_image[:, :, 1] = green_channel
blue_image[:, :, 2] = blue_channel

# Show the images
img = plt.imshow(A)
img.set_cmap('gray')
plt.axis('off')
plt.show()

img = plt.imshow(red_image)
img.set_cmap('gray')
plt.axis('off')
plt.show()

img = plt.imshow(green_image)
img.set_cmap('gray')
plt.axis('off')
plt.show()

img = plt.imshow(blue_image)
img.set_cmap('gray')
plt.axis('off')
plt.show()

# Combine again
X = np.dstack((red_channel, green_channel, blue_channel))
img = plt.imshow(X)
plt.axis('off')
plt.title('color')
plt.show()

# SVD for all colors
UR, SR, VTR = np.linalg.svd(red_channel, full_matrices=False)
SR = np.diag(SR)

UG, SG, VTG = np.linalg.svd(green_channel, full_matrices=False)
SG = np.diag(SG)

UB, SB, VTB = np.linalg.svd(blue_channel, full_matrices=False)
SB = np.diag(SB)

j = 0
for r in (20, 50):
    # Construct approximate image
    Rapprox = UR[:, :r] @ SR[0:r, :r] @ VTR[:r, :]
    Gapprox = UG[:, :r] @ SG[0:r, :r] @ VTG[:r, :]
    Bapprox = UB[:, :r] @ SB[0:r, :r] @ VTB[:r, :]

    Xapprox = np.stack((Rapprox, Gapprox, Bapprox), axis=-1)

    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r =' + str(r))
    plt.show()

print("done")

