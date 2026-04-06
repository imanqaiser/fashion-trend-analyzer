import pandas as pd
import requests
import cv2
import os
import time
import numpy as np


def download_images(df, batch_size, delay):
    """
    Download images in a batch with a delay between batches.
    """

    # make sure the images directory exists
    if not os.path.exists("../images/original_images"):
        os.makedirs("../images/original_images")

    # Iterate over the DataFrame rows with batch control
    for idx, row in df.iterrows():
        image_url = row["image"]
        id = row["id"]
        file_path = f"../images/original_images/{id}.jpg"

        # Check if the file already exists
        if os.path.exists(file_path):
            continue

        try:
            # Send a HTTP request
            response = requests.get(image_url)

            if response.status_code == 200:
                nparr = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                img = cv2.resize(img, (256, 256))
                cv2.imwrite(file_path, img)

                print(f"downloaded image {id}")

            # batch delay
            if (idx + 1) % batch_size == 0:
                time.sleep(delay)

        except Exception as e:
            print(f"Error for image {id}: {str(e)}")


# --- RUN SCRIPT DIRECTLY ---

df = pd.read_csv("../data/posts.csv")
download_images(df=df, batch_size=10, delay=10)
