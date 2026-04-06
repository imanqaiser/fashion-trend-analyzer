import os
import cv2
import numpy as np
import re
import json

from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_images_and_filenames(path):
    image_list = []
    filename_list = []

    print("Loading images from:", path)

    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(path, filename)

            img = cv2.imread(img_path)
            if img is None:
                print("Skipping bad image:", img_path)
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img / 255.0

            image_list.append(img)
            filename_list.append(filename)

    filename_list = [int(re.findall(r"\d+", s)[0]) for s in filename_list]

    X = np.array(image_list, dtype=np.float32)

    print("Loaded", len(X), "images")

    return X, filename_list


def create_encoder(path):
    print("Loading autoencoder from:", path)

    autoencoder = load_model(path)
    encoder = Model(
        inputs=autoencoder.input, outputs=autoencoder.get_layer("dropout_4").output
    )

    print("Encoder ready")

    return encoder


def get_feature_vectors(data, encoder):
    feature_vectors = []

    print("Generating feature vectors")

    for i, img in enumerate(data):
        img = np.expand_dims(img, axis=0)

        vector = encoder.predict(img, verbose=0)
        vector = np.reshape(vector, (-1))

        feature_vectors.append(vector)

        if (i + 1) % 50 == 0:
            print("Processed", i + 1, "images")

    feature_vectors = np.array(feature_vectors)

    print("Feature vector shape:", feature_vectors.shape)

    return feature_vectors


def scale_feature_vectors(feature_vectors):
    print("Scaling feature vectors")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_vectors)

    print("Scaling done")

    return scaled


def use_pca(feature_vectors, n_components=180):
    print("Running PCA with components:", n_components)

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(feature_vectors)

    print("PCA output shape:", reduced.shape)

    return reduced


def save_to_json(file_names, feature_vectors, path="../data/feature_vectors.json"):
    print("Saving to:", path)

    data = {"path": file_names, "feature_vectors": feature_vectors.tolist()}

    with open(path, "w") as f:
        json.dump(data, f)

    print("Saved")


# --- RUN ---

IMAGES_PATH = "../images/segmented_images/"
MODEL_PATH = "../models/autoencoder.h5"

X, file_names = get_images_and_filenames(IMAGES_PATH)

encoder = create_encoder(MODEL_PATH)

feature_vectors = get_feature_vectors(X, encoder)

scaled_feature_vectors = scale_feature_vectors(feature_vectors)

pca_feature_vectors = use_pca(scaled_feature_vectors)

save_to_json(file_names, pca_feature_vectors)

print("Done")
