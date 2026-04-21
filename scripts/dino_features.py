import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# -----------------------
# CONFIG
# -----------------------
IMG_DIR = "../images/original_images"
OUTPUT_PATH = "../data/dino_feature_vectors.json"
BATCH_SIZE = 32

os.makedirs("../data", exist_ok=True)

# -----------------------
# LOAD MODEL (DINOv2)
# -----------------------
print("Loading DINOv2 model...")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
model = model.to(device)
model.eval()

print("Model loaded")

# -----------------------
# TRANSFORMS (DINO style)
# -----------------------
transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


# -----------------------
# CENTER CROP FUNCTION (same as yours)
# -----------------------
def center_crop_clothing(img):
    w, h = img.size

    left = int(w * 0.2)
    right = int(w * 0.8)
    top = int(h * 0.15)
    bottom = int(h * 0.9)

    img = img.crop((left, top, right, bottom))
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

for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Extracting DINO features"):
    batch_files = image_files[i : i + BATCH_SIZE]
    batch_tensors = []
    batch_names = []

    for fname in batch_files:
        img_path = os.path.join(IMG_DIR, fname)
        try:
            img = Image.open(img_path).convert("RGB")

            # 🔥 same crop as CLIP
            img = center_crop_clothing(img)

            img = transform(img)

            batch_tensors.append(img)
            batch_names.append(fname)

        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            failed.append(fname)
            continue

    if not batch_tensors:
        continue

    batch_tensor = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        features = model(batch_tensor)

        # normalize (important)
        features = features / features.norm(dim=-1, keepdim=True)

    dino_feats = features.cpu().numpy()

    all_vectors.extend(dino_feats.tolist())
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
