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
OUTPUT_PATH = "../data/clip_color_feature_vectors.json"
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
# COLOR FEATURE FUNCTION
# -----------------------
def get_color_histogram(img, bins=16):
    img_np = np.array(img)

    hist_r = np.histogram(img_np[:, :, 0], bins=bins, range=(0, 255))[0]
    hist_g = np.histogram(img_np[:, :, 1], bins=bins, range=(0, 255))[0]
    hist_b = np.histogram(img_np[:, :, 2], bins=bins, range=(0, 255))[0]

    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
    hist = hist / (np.linalg.norm(hist) + 1e-8)
    return hist


# -----------------------
# CENTER CROP FUNCTION (KEY FIX)
# -----------------------
def center_crop_clothing(img):
    w, h = img.size

    # crop middle region (removes background)
    left = int(w * 0.2)
    right = int(w * 0.8)
    top = int(h * 0.15)
    bottom = int(h * 0.9)

    img = img.crop((left, top, right, bottom))

    # standardize size (reduces background detail)
    img = img.resize((224, 224))

    return img


# -----------------------
# GET IMAGE PATHS
# -----------------------
image_files = [
    f for f in os.listdir(IMG_DIR) if f.endswith(".jpg") or f.endswith(".png")
]
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
    batch_color_feats = []

    for fname in batch_files:
        img_path = os.path.join(IMG_DIR, fname)
        try:
            img = Image.open(img_path).convert("RGB")

            # 🔥 APPLY CROP HERE
            img = center_crop_clothing(img)

            batch_images.append(img)
            batch_names.append(fname)

            color_feat = get_color_histogram(img)
            batch_color_feats.append(color_feat)

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
        features = getattr(features, "pooler_output", features)
        features = features / features.norm(dim=-1, keepdim=True)

    clip_feats = features.cpu().numpy()
    color_feats = np.array(batch_color_feats)

    # -----------------------
    # COMBINE FEATURES
    # -----------------------
    combined = np.concatenate(
        [
            clip_feats * 0.7,  # slightly favor structure
            color_feats * 0.3,  # keep color but not dominant
        ],
        axis=1,
    )

    all_vectors.extend(combined.tolist())
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
