import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def get_image_paths(image_dir):
    print("Loading images from:", image_dir)

    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")
    ]

    print("Found", len(image_paths), "images")
    return image_paths


def segment_image(yolo, mask_predictor, image_path):
    print("-----------------------------")
    print("Processing image:", image_path)

    print("Running YOLO detection")
    yolo_output = yolo.predict(image_path, conf=0.5)

    boxes = []
    for result in yolo_output:
        if result.boxes is None:
            continue

        for bbox in result.boxes.data:
            b = bbox.int().cpu().numpy()
            boxes.append([b[0], b[1], b[2], b[3]])

    print("Detected", len(boxes), "bounding boxes")

    if len(boxes) == 0:
        print("No detections, skipping")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    output = np.zeros_like(image)

    for i, box in enumerate(boxes):
        print("Processing box", i + 1, ":", box)

        box = np.array(box)

        print("Running SAM")
        mask_predictor.set_image(image)

        masks, scores, logits = mask_predictor.predict(box=box, multimask_output=True)

        print("SAM returned", len(masks), "masks")
        print("Scores:", scores)

        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]

        print("Using mask", best_idx, "with score", scores[best_idx])

        mask_combined = np.logical_or(mask_combined, best_mask)

    print("Applying combined mask")
    output[mask_combined] = image[mask_combined]

    save_path = os.path.join(
        OUTPUT_DIR, "outfit_" + os.path.basename(image_path).replace(".jpg", ".png")
    )

    success = cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    if success:
        print("Saved segmented image:", save_path)
    else:
        print("Failed to save image:", save_path)


MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "../models/sam_weights.pth"
YOLO_WEIGHTS = "../models/yolo_weights.pt"
IMAGE_DIR = "../images/original_images"
OUTPUT_DIR = "../images/segmented_images"

print("Starting segmentation pipeline")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Output directory ready:", OUTPUT_DIR)

print("Loading YOLO model")
yolo = YOLO(YOLO_WEIGHTS)
print("YOLO loaded")

print("Loading SAM model")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(get_device())
mask_predictor = SamPredictor(sam)
print("SAM loaded")

image_paths = get_image_paths(IMAGE_DIR)

for image_path in image_paths:
    save_path = os.path.join(
        OUTPUT_DIR, "outfit_" + os.path.basename(image_path).replace(".jpg", ".png")
    )

    if os.path.exists(save_path):
        print("Skipping existing file:", save_path)
        continue

    segment_image(yolo, mask_predictor, image_path)

print("All images processed")
