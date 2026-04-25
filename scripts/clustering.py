import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans

# -----------------------
# CONFIG
# -----------------------
PCA_PATH = "../data/pca_vectors.json"
OUTPUT_PATH = "../data/clip_feature_vectors_clustered.json"

K_RANGE = range(2, 11)

# -----------------------
# LOAD PCA VECTORS
# -----------------------
print("Loading PCA vectors...")
with open(PCA_PATH) as f:
    data = json.load(f)

vectors = np.array(data["pca_vectors"], dtype=np.float32)
paths = np.array(data["paths"])

print(f"Loaded {len(paths)} points")

# -----------------------
# ELBOW (KMEANS INERTIA)
# -----------------------
print("\nRunning elbow method...")

inertias = []

for k in K_RANGE:
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        random_state=42,
    )
    labels = kmeans.fit_predict(vectors)

    inertias.append(kmeans.inertia_)


# -----------------------
# PLOT ELBOW
# -----------------------
plt.figure(figsize=(6, 4))
plt.plot(list(K_RANGE), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method (KMeans)")
plt.grid()
plt.show()
# -----------------------
# FINAL CLUSTERING
# -----------------------
FINAL_K = 7
print(f"\nRunning clustering with k={FINAL_K}...")

# KMeans++
kmeans = KMeans(
    n_clusters=FINAL_K,
    init="k-means++",
    n_init=20,
    random_state=42,
)
labels_kmeans = kmeans.fit_predict(vectors)


# -----------------------
# STATS
# -----------------------
def print_stats(name, labels):
    unique = sorted(set(labels))
    print(f"\n{name}: {len(unique)} clusters")
    for c in unique:
        count = np.sum(labels == c)
        print(f"  Cluster {c}: {count} images")


print_stats("KMeans++", labels_kmeans)

# -----------------------
# SAVE
# -----------------------
output = {
    "paths": paths.tolist(),
    "pca_vectors": vectors.tolist(),
    "pca_2d": data["pca_2d"],
    "cluster": labels_kmeans.tolist(),
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f)

print(f"\nSaved to {OUTPUT_PATH}")
