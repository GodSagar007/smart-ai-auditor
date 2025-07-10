import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Generate sample linear data
np.random.seed(42)
X = np.linspace(0, 10, 50)
true_w = 2.5
true_b = 5
y = true_w * X + true_b + np.random.randn(*X.shape) * 1.0  # small noise

# 2. Mean Squared Error
def mse(w, b):
    y_pred = w * X + b
    return np.mean((y - y_pred) ** 2)

# 3. Gradients
def gradients(w, b):
    y_pred = w * X + b
    dw = -2 * np.mean(X * (y - y_pred))
    db = -2 * np.mean(y - y_pred)
    return dw, db

# 4. Gradient Descent
w, b = 0.0, 0.0
lr = 0.005
steps = 60
w_hist, b_hist, loss_hist = [], [], []

for _ in range(steps):
    loss = mse(w, b)
    w_hist.append(w)
    b_hist.append(b)
    loss_hist.append(loss)
    dw, db = gradients(w, b)
    w -= lr * dw
    b -= lr * db

# 5. Create MSE surface
w_vals = np.linspace(0, 5, 100)
b_vals = np.linspace(0, 10, 100)
W, B = np.meshgrid(w_vals, b_vals)
Z = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = mse(W[i, j], B[i, j])

# 6. Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(W, B, Z, cmap='plasma', alpha=0.9, edgecolor='none')

# Gradient descent path
ax.plot(w_hist, b_hist, loss_hist, color='red', marker='o', label='Gradient Descent Path')

# Final point
ax.scatter(w_hist[-1], b_hist[-1], loss_hist[-1], color='yellow', s=60, marker='x', label='Final Minimum')
ax.text(w_hist[-1], b_hist[-1], loss_hist[-1], "  ← Minimum", color='black', fontsize=10)

# Labels and view
ax.set_xlabel("Weight (w)", fontsize=12)
ax.set_ylabel("Bias (b)", fontsize=12)
ax.set_zlabel("Loss (MSE)", fontsize=12)
ax.set_title("Gradient Descent on MSE Bowl Surface", fontsize=14)

# 👇 Perfect side angle view (like looking into the bowl)
ax.view_init(elev=35, azim=135)

fig.colorbar(surf, shrink=0.5, aspect=10, label="MSE")
ax.legend()
plt.tight_layout()
plt.show()
