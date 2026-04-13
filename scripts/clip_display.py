import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import matplotlib.pyplot as plt
import math

# -----------------------
# CONFIG
# -----------------------
VECTORS_PATH = "../data/clip_feature_vectors.json"
IMG_DIR = "../images/original_images"
MODEL_NAME = "openai/clip-vit-base-patch32"

query = "soft white flowing ethereal angelic aesthetic, sheer fabric, delicate, heavenly, dreamy pastel"  # <-- change this

# -----------------------
# LOAD MODEL
# -----------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

# -----------------------
# LOAD PRECOMPUTED VECTORS
# -----------------------
with open(VECTORS_PATH) as f:
    data = json.load(f)

image_vectors = torch.tensor(data["feature_vectors"]).to(device)  # (N, 512)
image_paths = data["paths"]

# -----------------------
# QUERY
# -----------------------
with torch.no_grad():
    text_inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)

    text_features = model.get_text_features(**text_inputs)

    # handle weird HF return types (same issue as before)
    if not isinstance(text_features, torch.Tensor):
        text_features = getattr(text_features, "pooler_output", text_features)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# cosine similarity (vectors already normalized)
similarities = (image_vectors @ text_features.T).squeeze()  # (N,)


# -----------------------
# SHOW ALL RESULTS (ONE FIGURE)
# -----------------------
print(f"\nQuery: '{query}'")
print(f"Total images: {len(image_paths)}, showing all in one figure\n")

all_indices = similarities.argsort(descending=True)

n = len(all_indices)
cols = 5  # you can tweak this
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
axes = axes.flatten()


for i, idx in enumerate(all_indices):
    fname = image_paths[idx]
    score = similarities[idx].item()

    img = Image.open(os.path.join(IMG_DIR, fname))
    axes[i].imshow(img)
    axes[i].set_title(f"#{i + 1}\n{score:.3f}", fontsize=7)
    axes[i].axis("off")

# hide extra empty slots
for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
