import json
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
VECTORS_PATH = "../data/clip_color_feature_vectors.json"
OUTPUT_PATH = "../data/pca_vectors.json"


# -----------------------
# PCA (FULL SVD)
# -----------------------
def fit_pca_full(vectors):
    centered = vectors - vectors.mean(axis=0, keepdims=True)

    # SVD
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    explained_variance = (singular_values**2) / max(len(vectors) - 1, 1)
    total_variance = explained_variance.sum()
    explained_ratio = explained_variance / total_variance

    cumulative_variance = np.cumsum(explained_ratio)

    return vt, explained_ratio, cumulative_variance, centered


# -----------------------
# LOAD FEATURES
# -----------------------
print("Loading feature vectors...")
with open(VECTORS_PATH) as f:
    data = json.load(f)

vectors = np.array(data["feature_vectors"], dtype=np.float32)
paths = data["paths"]

print(f"Loaded {len(vectors)} vectors of dim {vectors.shape[1]}")

# -----------------------
# FIT PCA (FULL)
# -----------------------
print("Fitting PCA (full spectrum)...")
vt, explained_ratio, cumulative_variance, centered = fit_pca_full(vectors)

print("PCA fitted")

# -----------------------
# PLOT ELBOW (CUMULATIVE)
# -----------------------
plt.figure(figsize=(6, 4))
plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance Curve")
plt.grid()
plt.show()

# -----------------------
# PLOT SCREE
# -----------------------
plt.figure(figsize=(6, 4))
plt.plot(explained_ratio)
plt.xlabel("Component Index")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid()
plt.show()

# -----------------------
# SELECT K
# -----------------------
k = 50

# -----------------------
# PROJECT TO PCA (k dims)
# -----------------------
components_k = vt[:k]
pca_vectors = centered @ components_k.T

print(f"PCA ({k}D) shape: {pca_vectors.shape}")

# -----------------------
# PROJECT TO PCA (2D)
# -----------------------
components_2d = vt[:2]
pca_2d = centered @ components_2d.T

print(f"PCA (2D) shape: {pca_2d.shape}")

# -----------------------
# SAVE
# -----------------------
output = {
    "paths": paths,
    "pca_vectors": pca_vectors.astype(np.float32).tolist(),
    "pca_2d": pca_2d.astype(np.float32).tolist(),
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f)

print(f"Saved to {OUTPUT_PATH}")
