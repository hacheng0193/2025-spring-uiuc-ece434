import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------------
# 1. Define a 3x3 matrix A (not necessarily symmetric)
# ----------------------------------------------------
A = np.array([
    [3, 2, 4],
    [2, 0, 2],
    [4, 2, 3]
], dtype=float)

# ----------------------------------------------------
# 2. Eigen-decomposition (general)
#    np.linalg.eig works for any square matrix
# ----------------------------------------------------
eigenvalues, eigenvectors = np.linalg.eig(A)
Q = eigenvectors / np.linalg.norm(eigenvectors, axis=0)  # Normalize eigenvectors
Q_inv = np.linalg.inv(Q)  # For non-orthonormal eigenvectors, Q_inv != Q^T

# ----------------------------------------------------
# 3. Points to plot in original (x,y,z) space
# ----------------------------------------------------
# Define the points
points = np.array([[1, 0, -1],
                  [2, 1, -1],
                  [-1, 2, 0],
                  [3, 0, 4],
                  [2, 1, -2],
                  [1, -1, -2],
                  [-3, 1, 1],
                  [0, -4, 1],
                  [-1, 3, 1],
                  [1, 0, 1]])

# ----------------------------------------------------
# 4. Transform each point into the new basis
#    p_new = Q_inv * p
# ----------------------------------------------------
points_new = points @ Q_inv

# ----------------------------------------------------
# 5. Plot both sets of points
# ----------------------------------------------------
fig = plt.figure(figsize=(12,5))

# --- 5A. Plot in the original coordinates ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(points[:,0], points[:,1], points[:,2], c='b', marker='o')
ax1.set_title("Points in Original Coordinates")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.view_init(elev=20, azim=35) 
# Annotate each point with its coordinates
for i in range(points.shape[0]):
    ax1.text(points[i, 0], points[i, 1], points[i, 2], f'({points[i, 0]}, {points[i, 1]}, {points[i, 2]})')

# --- 5B. Plot in the new (eigenvector) basis ---
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(points_new[:,0], points_new[:,1], points_new[:,2], c='r', marker='^')
ax2.set_title("Points in the Eigenvector Basis")
ax2.set_xlabel(f"x_new ({Q[:,0][0]:.4f}, {Q[:,0][1]:.4f}, {Q[:,0][2]:.4f})")
ax2.set_ylabel(f"y_new ({Q[:,1][0]:.4f}, {Q[:,1][1]:.4f}, {Q[:,1][2]:.4f})")
ax2.set_zlabel(f"z_new ({Q[:,2][0]:.4f}, {Q[:,2][1]:.4f}, {Q[:,2][2]:.4f})")
for i in range(points_new.shape[0]):
    ax2.text(points_new[i, 0], points_new[i, 1], points_new[i, 2], 
             f'({points_new[i, 0]:.4f}, {points_new[i, 1]:.4f}, {points_new[i, 2]:.4f})')
# ----------------------------------------------------
# 6. Mark the basis vectors in the new coordinate plot
#    In the new basis, e1 = (1,0,0), e2 = (0,1,0), e3 = (0,0,1).
# ----------------------------------------------------
# We'll draw arrows (quivers) from origin to each basis vector,
# and label them.  These appear "orthogonal" in the second plot
# because that plot is just a standard 3D coordinate system for
# (x_new, y_new, z_new).
ax2.view_init(elev=20, azim=35)  # adjust as you like

plt.tight_layout()
plt.show()
