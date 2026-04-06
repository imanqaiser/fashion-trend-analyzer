from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import json


NUMBER_OF_CLUSTERS = 7


def print_cluster_counts(labels):
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print("  Cluster", cluster, "->", count, "images")


def load_data():
    with open("../data/feature_vectors.json", "r") as f:
        data = json.load(f)
    feature_vector = np.array(data["feature_vectors"])
    return data, feature_vector


def run_kmeans(feature_vector, n_clusters):
    """Standard KMeans with random init."""
    model = KMeans(n_clusters=n_clusters, init="random", random_state=0, n_init=10)
    labels = model.fit_predict(feature_vector)
    score = silhouette_score(feature_vector, labels)
    print("[KMeans]           Silhouette score:", round(score, 4))
    print_cluster_counts(labels)
    return labels, score


def run_kmeans_plus(feature_vector, n_clusters):
    """KMeans++ with smarter centroid initialization (usually converges better)."""
    model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0, n_init=10)
    labels = model.fit_predict(feature_vector)
    score = silhouette_score(feature_vector, labels)
    print("[KMeans++]         Silhouette score:", round(score, 4))
    print_cluster_counts(labels)
    return labels, score


def run_bisecting_kmeans(feature_vector, n_clusters):
    """Bisecting KMeans — recursively splits clusters, tends to produce more balanced groups."""
    model = BisectingKMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(feature_vector)
    score = silhouette_score(feature_vector, labels)
    print("[Bisecting KMeans] Silhouette score:", round(score, 4))
    print_cluster_counts(labels)
    return labels, score


def run_hierarchical_ward(feature_vector, n_clusters):
    """Agglomerative clustering with Ward linkage — minimizes within-cluster variance."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(feature_vector)
    score = silhouette_score(feature_vector, labels)
    print("[Hierarchical Ward]     Silhouette score:", round(score, 4))
    print_cluster_counts(labels)
    return labels, score


def run_hierarchical_complete(feature_vector, n_clusters):
    """Agglomerative clustering with complete linkage — merges based on max pairwise distance."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete")
    labels = model.fit_predict(feature_vector)
    score = silhouette_score(feature_vector, labels)
    print("[Hierarchical Complete] Silhouette score:", round(score, 4))
    print_cluster_counts(labels)
    return labels, score


def run_hierarchical_average(feature_vector, n_clusters):
    """Agglomerative clustering with average linkage — merges based on average pairwise distance."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
    labels = model.fit_predict(feature_vector)
    score = silhouette_score(feature_vector, labels)
    print("[Hierarchical Average]  Silhouette score:", round(score, 4))
    print_cluster_counts(labels)
    return labels, score


data, feature_vector = load_data()

print("\nRunning clustering with", NUMBER_OF_CLUSTERS, "clusters...\n")

results = {
    "kmeans": run_kmeans(feature_vector, NUMBER_OF_CLUSTERS),
    "kmeans_plus": run_kmeans_plus(feature_vector, NUMBER_OF_CLUSTERS),
    "bisecting_kmeans": run_bisecting_kmeans(feature_vector, NUMBER_OF_CLUSTERS),
    "hierarchical_ward": run_hierarchical_ward(feature_vector, NUMBER_OF_CLUSTERS),
    "hierarchical_complete": run_hierarchical_complete(
        feature_vector, NUMBER_OF_CLUSTERS
    ),
    "hierarchical_average": run_hierarchical_average(
        feature_vector, NUMBER_OF_CLUSTERS
    ),
}

# Pick best method by silhouette score
best_method = max(results, key=lambda k: results[k][1])
best_labels, best_score = results[best_method]

print("\nBest method:", best_method, "(silhouette =", round(best_score, 4), ")")

kmeans_pp_labels, _ = results["kmeans_plus"]

data["cluster"] = kmeans_pp_labels.tolist()

with open("../data/feature_vectors_clustered.json", "w") as f:
    json.dump(data, f)

print("Saved KMeans++ clusters only")
