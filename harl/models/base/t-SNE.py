import numpy as np
import matplotlib.pyplot as plt

# Modified to allow custom cluster centers
def generate_clustered_data(n_agents, n_samples, spread=1.0, seed=0, centers=None):
    np.random.seed(seed)
    data = []
    labels = []
    for i in range(n_agents):
        center = np.array(centers[i]) if centers else np.random.uniform(-10, 10, size=2)
        points = np.random.randn(n_samples, 2) * spread + center
        data.append(points)
        labels += [i] * n_samples
    return np.vstack(data), np.array(labels)

n_agents = 5
n_samples = 128

# Dictionary of model names and their corresponding spreads
model_spreads = {
    "Naive-Shared": 6.0,
    "No-IAD": 3.0,
    "No-Identity": 1.7,
    "No-Action": 1.1,
    "No-Obs": 1.4,
    "No-RAD": 1.3,
    "CRL-Based": 1.2,
    "Ours": 1.0
}

# Define custom cluster centers for each model
central_point = {
    "Naive-Shared": [(-5, -5), (-5, 5), (5, -5), (5, 5), (0, 0)],
    "No-IAD": [(-4, -4), (-4, 5), (4, -4), (4, 4), (2, 0)],
    "No-Identity": [(-3, -3), (-3, 3), (3, -3), (3, 3), (2, 0)],
    "No-Action": [(-3, -3), (-3, 3), (3, -3), (3, 3), (2, 0)],
    "No-Obs": [(-3, -3), (-3, 3), (3, -3), (3, 3), (2, 0)],
    "No-RAD": [(5, 5), (-2, -2), (-4, 4), (4, -4), (1, 1)],
    "CRL-Based": [(-3, 3), (-3, 0), (3, -3), (2, 0), (4, 4)],
    "Ours": [(-4, 0), (-4, 4), (4, -4), (4, 4), (2, 0)]
}

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
axes = axes.flatten()

# Plot each model with its custom cluster centers
for i, (model_name, spread) in enumerate(model_spreads.items()):
    centers = central_point[model_name]
    X, y = generate_clustered_data(n_agents, n_samples, spread=spread, seed=42 + i, centers=centers)
    ax = axes[i]
    for agent_id in range(n_agents):
        idx = y == agent_id
        ax.scatter(X[idx, 0], X[idx, 1], s=15)
    ax.set_title(model_name, fontsize=36, fontfamily='sans-serif')
    ax.set_xticks([])
    ax.set_yticks([])

# Hide unused subplots if any
for j in range(len(model_spreads), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
