from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [16, 8]

# Load image
A = imread('comet_crop.jpg')
X = np.mean(A, -1)  # Convert RGB to grayscale
print(X)  # Print the image matrix


img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.show()

# SVD
U, S, VT = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)
print("U:", U)  # Print the SVD components
print("S:", S)
print("VT:", VT)

print(len(A), len(A[0]))  # Find sizes of matrices
print(len(U), len(U[0]))
print(len(S), len(S[0]))
print(len(VT), len(VT[0]))

j = 0
for r in (1, 2, 3, 5, 20, 100):
    # Construct approximate image
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    plt.figure(j+1)
    j += 1
    # Display image
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r =' + str(r))
    plt.show()
    print("r=", r, ": ")  # Find sizes of Xapprox
    print("U:", len(U[:, :r]), " x ", len(U[:, :r][0]))
    print("S:", len(S[0:r, :r]), " x ", len(S[0:r, :r][0]))
    print("VT:", len(VT[:r, :]), " x ", len(VT[:r, :][0]))

print("done")
