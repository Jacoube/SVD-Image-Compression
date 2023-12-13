from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16, 8]

A = imread('comet_crop.jpg')
X = np.mean(A, -1)  # Convert RGB to grayscale

img = plt.imshow(A)
img.set_cmap('gray')
plt.axis('off')
plt.show()

U, S, VT = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)

j = 0
for r in (1, 2, 3, 5, 20, 100):
    # Construct approximate image
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r =' + str(r))
    plt.show()

print("done")
