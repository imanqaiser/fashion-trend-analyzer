import json
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, BisectingKMeans
from sklearn.metrics import silhouette_score
import warnings

# -----------------------
# CONFIG
# -----------------------
PCA_PATH = "../data/pca_dinovectors.json"
FINAL_K = 7

# -----------------------
# LOAD DATA
# -----------------------
print("Loading PCA vectors...")
with open(PCA_PATH) as f:
    data = json.load(f)

vectors = np.array(data["pca_vectors"], dtype=np.float32)
paths = np.array(data["paths"])

print(f"Loaded {len(paths)} samples")

# -----------------------
# OPTIONAL HDBSCAN
# -----------------------
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except:
    HDBSCAN_AVAILABLE = False


# -----------------------
# EVALUATION FUNCTION
# -----------------------
def evaluate(name, labels):
    unique = set(labels)

    # remove noise label if exists
    valid_clusters = [c for c in unique if c != -1]

    print(f"\n===== {name} =====")
    print(f"Clusters (excluding noise): {len(valid_clusters)}")

    for c in sorted(unique):
        count = np.sum(labels == c)
        print(f"Cluster {c}: {count} images")

    # silhouette
    if len(valid_clusters) > 1:
        try:
            score = silhouette_score(vectors, labels)
            print(f"Silhouette Score: {score:.4f}")
        except:
            print("Silhouette Score: could not compute")
    else:
        print("Silhouette Score: not applicable")


# -----------------------
# RUN METHODS
# -----------------------

print("\nRunning clustering comparisons...")

# KMeans++
kmeans_pp = KMeans(n_clusters=FINAL_K, init="k-means++", n_init=20, random_state=42)
labels_kpp = kmeans_pp.fit_predict(vectors)
evaluate("KMeans++", labels_kpp)

# KMeans (random)
kmeans_rand = KMeans(n_clusters=FINAL_K, init="random", n_init=20, random_state=42)
labels_krand = kmeans_rand.fit_predict(vectors)
evaluate("KMeans (random)", labels_krand)

# Bisecting KMeans
bisect = BisectingKMeans(n_clusters=FINAL_K, random_state=42)
labels_bisect = bisect.fit_predict(vectors)
evaluate("Bisecting KMeans", labels_bisect)

# Agglomerative (Ward)
agg_ward = AgglomerativeClustering(n_clusters=FINAL_K, linkage="ward")
labels_ward = agg_ward.fit_predict(vectors)
evaluate("Agglomerative (Ward)", labels_ward)

# Agglomerative (Average)
agg_avg = AgglomerativeClustering(n_clusters=FINAL_K, linkage="average")
labels_avg = agg_avg.fit_predict(vectors)
evaluate("Agglomerative (Average)", labels_avg)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(vectors)
evaluate("DBSCAN", labels_db)

# HDBSCAN
if HDBSCAN_AVAILABLE:
    hdb = hdbscan.HDBSCAN(min_cluster_size=5)
    labels_hdb = hdb.fit_predict(vectors)
    evaluate("HDBSCAN", labels_hdb)
else:
    print("\nHDBSCAN not installed (pip install hdbscan)")

print("\nDone.")
