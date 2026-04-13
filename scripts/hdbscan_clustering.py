import json
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import os

# -----------------------
# CONFIG
# -----------------------
UMAP_PATH = "../data/umap_vectors.json"
OUTPUT_PATH = "../data/clip_feature_vectors_clustered.json"

# -----------------------
# LOAD UMAP VECTORS
# -----------------------
print("Loading UMAP vectors...")
with open(UMAP_PATH) as f:
    data = json.load(f)

vectors_15d = np.array(data["umap_15d"], dtype=np.float32)
vectors_2d = np.array(data["umap_2d"], dtype=np.float32)
paths = data["paths"]

print(f"Loaded {len(paths)} points")

# -----------------------
# HDBSCAN
# -----------------------
print("Running HDBSCAN...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,  # minimum images per cluster — tune this
    min_samples=3,
    metric="euclidean",
    cluster_selection_method="eom",
)

labels = clusterer.fit_predict(vectors_15d)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = sum(1 for l in labels if l == -1)

print(f"\nFound {n_clusters} clusters")
print(f"Noise points (unclustered): {n_noise} ({n_noise / len(labels) * 100:.1f}%)")
print(f"\nCluster sizes:")
for c in sorted(set(labels)):
    count = sum(1 for l in labels if l == c)
    label = "NOISE" if c == -1 else f"Cluster {c}"
    print(f"  {label}: {count} images")

# -----------------------
# VISUALIZE
# -----------------------
plt.figure(figsize=(12, 8))

unique_labels = sorted(set(labels))
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    mask = labels == label
    if label == -1:
        plt.scatter(
            vectors_2d[mask, 0],
            vectors_2d[mask, 1],
            c="lightgray",
            s=10,
            alpha=0.4,
            label="noise",
        )
    else:
        plt.scatter(
            vectors_2d[mask, 0],
            vectors_2d[mask, 1],
            c=[color],
            s=20,
            alpha=0.7,
            label=f"cluster {label} ({mask.sum()})",
        )

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.title(f"HDBSCAN Clustering — {n_clusters} clusters, {n_noise} noise points")
plt.tight_layout()
plt.savefig("../data/clusters_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to ../data/clusters_plot.png")

# -----------------------
# SAVE
# -----------------------
output = {
    "paths": paths,
    "umap_15d": data["umap_15d"],
    "umap_2d": data["umap_2d"],
    "cluster": labels.tolist(),
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f)

print(f"\nSaved to {OUTPUT_PATH}")
