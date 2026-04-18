import json
import numpy as np
import os
from PIL import Image
import math
import tensorflow as tf
from tensorboard.plugins import projector

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "../data/clip_feature_vectors_clustered.json"
IMG_DIR = "../images/original_images"
OUTPUT_DIR = "../data/tensorboard"
THUMB_SIZE = (64, 64)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
print("Loading data...")
with open(DATA_PATH) as f:
    data = json.load(f)

vectors = np.array(data["pca_50d"], dtype=np.float32)
clusters = data["cluster"]
paths = data["paths"]

print(
    f"Loaded {len(paths)} points, {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters"
)

# -----------------------
# SPRITE
# -----------------------
print("Building sprite...")
images = []
valid_indices = []

for i, fname in enumerate(paths):
    img_path = os.path.join(IMG_DIR, fname)
    if not os.path.exists(img_path):
        print(f"Missing: {img_path}")
        continue
    img = Image.open(img_path).convert("RGB").resize(THUMB_SIZE)
    images.append(img)
    valid_indices.append(i)

# filter vectors/clusters/paths to only valid images
vectors = vectors[valid_indices]
clusters = [clusters[i] for i in valid_indices]
paths = [paths[i] for i in valid_indices]

n = len(images)
grid_size = math.ceil(math.sqrt(n))
sprite = Image.new(
    "RGB", (THUMB_SIZE[0] * grid_size, THUMB_SIZE[1] * grid_size), (255, 255, 255)
)
for i, img in enumerate(images):
    row, col = divmod(i, grid_size)
    sprite.paste(img, (col * THUMB_SIZE[0], row * THUMB_SIZE[1]))

sprite.save(os.path.join(OUTPUT_DIR, "sprite.png"))
print(f"Sprite saved: {n} images in {grid_size}x{grid_size} grid")

# -----------------------
# METADATA
# -----------------------
print("Writing metadata...")
with open(os.path.join(OUTPUT_DIR, "metadata.tsv"), "w") as f:
    f.write("filename\tcluster\n")
    for fname, cluster in zip(paths, clusters):
        label = "noise" if cluster == -1 else f"cluster_{cluster}"
        f.write(f"{fname}\t{label}\n")

# -----------------------
# EMBEDDINGS + PROJECTOR CONFIG
# -----------------------
print("Saving embeddings...")
embedding_var = tf.Variable(vectors, name="fashion_embeddings")
checkpoint = tf.train.Checkpoint(embedding=embedding_var)
checkpoint.save(os.path.join(OUTPUT_DIR, "embedding.ckpt"))

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = "metadata.tsv"
embedding.sprite.image_path = "sprite.png"
embedding.sprite.single_image_dim.extend(THUMB_SIZE)
projector.visualize_embeddings(OUTPUT_DIR, config)

print("\nDone! Now run:")
print(f'tensorboard --logdir "{OUTPUT_DIR}"')
