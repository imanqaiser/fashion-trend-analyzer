import json
import numpy as np
import umap
import os

# -----------------------
# CONFIG
# -----------------------
VECTORS_PATH = "../data/clip_feature_vectors.json"
OUTPUT_PATH = "../data/umap_vectors.json"

# -----------------------
# LOAD CLIP VECTORS
# -----------------------
print("Loading CLIP vectors...")
with open(VECTORS_PATH) as f:
    data = json.load(f)

vectors = np.array(data["feature_vectors"], dtype=np.float32)
paths = data["paths"]

print(f"Loaded {len(vectors)} vectors of dim {vectors.shape[1]}")

# -----------------------
# FIT UMAP
# -----------------------
print("Fitting UMAP...")

reducer = umap.UMAP(
    n_components=15,  # reduce to 15 dims for clustering
    n_neighbors=15,  # how many neighbors to consider (higher = more global structure)
    min_dist=0.0,  # 0.0 is best for clustering (tighter clusters)
    metric="cosine",  # cosine is best for CLIP embeddings
    random_state=42,
    verbose=True,
)

reduced = reducer.fit_transform(vectors)

print(f"UMAP done. Output shape: {reduced.shape}")

# -----------------------
# ALSO SAVE 2D FOR VISUALIZATION
# -----------------------
print("Fitting 2D UMAP for visualization...")

reducer_2d = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,  # slightly higher for nicer visualization
    metric="cosine",
    random_state=42,
    verbose=True,
)

reduced_2d = reducer_2d.fit_transform(vectors)

print(f"2D UMAP done. Output shape: {reduced_2d.shape}")

# -----------------------
# SAVE
# -----------------------
output = {
    "paths": paths,
    "umap_15d": reduced.tolist(),
    "umap_2d": reduced_2d.tolist(),
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f)

print(f"Saved to {OUTPUT_PATH}")
