import os
import hashlib
import shutil

# -----------------------
# CONFIG
# -----------------------
IMG_DIR = "../images/original_images"
OUTPUT_DIR = "../images/deduplicated_images"  # where clean images go
MOVE_DUPLICATES = False  # True = move duplicates, False = just skip them
DUPLICATE_DIR = "../images/duplicates"  # only used if MOVE_DUPLICATES = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
if MOVE_DUPLICATES:
    os.makedirs(DUPLICATE_DIR, exist_ok=True)


# -----------------------
# HASH FUNCTION
# -----------------------
def get_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# -----------------------
# DEDUPLICATION
# -----------------------
image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")]
image_files.sort()

seen_hashes = {}
unique_count = 0
duplicate_count = 0

for fname in image_files:
    src_path = os.path.join(IMG_DIR, fname)
    file_hash = get_hash(src_path)

    if file_hash not in seen_hashes:
        # keep this image
        seen_hashes[file_hash] = fname
        dst_path = os.path.join(OUTPUT_DIR, fname)
        shutil.copy2(src_path, dst_path)
        unique_count += 1
    else:
        duplicate_count += 1

        if MOVE_DUPLICATES:
            dst_path = os.path.join(DUPLICATE_DIR, fname)
            shutil.move(src_path, dst_path)

        print(f"Duplicate: {fname} == {seen_hashes[file_hash]}")

# -----------------------
# SUMMARY
# -----------------------
print("\nDone!")
print(f"Total images processed: {len(image_files)}")
print(f"Unique images kept: {unique_count}")
print(f"Duplicates found: {duplicate_count}")
