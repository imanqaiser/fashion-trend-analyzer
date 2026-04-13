import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
IMG_DIR = "../images/original_images"
OUTPUT_PATH = "../data/clip_feature_vectors.json"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32

os.makedirs("../data", exist_ok=True)


# -----------------------
# LOAD MODEL
# -----------------------
print("Loading CLIP model...")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded")

# -----------------------
# GET IMAGE PATHS
# -----------------------
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
image_files.sort()
print(f"Found {len(image_files)} images")

# -----------------------
# EXTRACT FEATURES
# -----------------------
all_vectors = []
all_paths = []
failed = []

for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Extracting features"):
    batch_files = image_files[i : i + BATCH_SIZE]
    batch_images = []
    batch_names = []

    for fname in batch_files:
        img_path = os.path.join(IMG_DIR, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            batch_images.append(img)
            batch_names.append(fname)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            failed.append(fname)
            continue

    if not batch_images:
        continue

    with torch.no_grad():
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(
            device
        )

        features = model.get_image_features(pixel_values=inputs["pixel_values"])
        # always grab embedding
        features = getattr(features, "pooler_output", features)
        features = features / features.norm(dim=-1, keepdim=True)

    all_vectors.extend(features.cpu().numpy().tolist())
    all_paths.extend(batch_names)

print(f"\nExtracted features for {len(all_vectors)} images")
if failed:
    print(f"Failed: {len(failed)} images")

# -----------------------
# SAVE
# -----------------------
output = {"paths": all_paths, "feature_vectors": all_vectors}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f)

print(f"Saved to {OUTPUT_PATH}")
