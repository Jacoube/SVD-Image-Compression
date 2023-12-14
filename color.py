from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [16, 8]

# Load image
A = imread('comet_crop.jpg')
print(A)  # Print the image matrix

# Show original image
img = plt.imshow(A)
plt.axis('off')
plt.show()

# Extract individual color channels
red_channel = A[:, :, 0]
green_channel = A[:, :, 1]
blue_channel = A[:, :, 2]

# Perform SVD on each color channel
UR, SR, VTR = np.linalg.svd(red_channel, full_matrices=False)
SR = np.diag(SR)

UG, SG, VTG = np.linalg.svd(green_channel, full_matrices=False)
SG = np.diag(SG)

UB, SB, VTB = np.linalg.svd(blue_channel, full_matrices=False)
SB = np.diag(SB)

j = 0
for r in (1, 2, 3, 5, 20, 100, 500):
    # Construct approximate images for each color channel
    Rapprox = UR[:, :r] @ SR[0:r, :r] @ VTR[:r, :]
    Gapprox = UG[:, :r] @ SG[0:r, :r] @ VTG[:r, :]
    Bapprox = UB[:, :r] @ SB[0:r, :r] @ VTB[:r, :]

    # Normalize the reconstructed channels to the [0, 255] range
    Rapprox = np.clip(Rapprox, 0, 255).astype(np.uint8)
    Gapprox = np.clip(Gapprox, 0, 255).astype(np.uint8)
    Bapprox = np.clip(Bapprox, 0, 255).astype(np.uint8)

    # Merge the color channels back into RGB image
    Xapprox = np.dstack((Rapprox, Gapprox, Bapprox))

    # Display reconstructed images
    plt.imshow(Xapprox)
    plt.axis('off')
    plt.title('r =' + str(r))
    plt.show()

plt.figure(1)  # Plot of singular values
plt.semilogy(np.diag(SR))
plt.title('Singular Values')
plt.show()

plt.figure(2)  # Cumulative sum of singular values
plt.semilogy(np.cumsum(np.diag(SR))/np.sum(np.diag(SR)))
plt.title('Singular Values: Cumulative Sum')
plt.show()
