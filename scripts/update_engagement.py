import json
import random
from datetime import datetime, timedelta

# -----------------------
# COMMENT TEMPLATES (UPDATED)
# -----------------------
COMMENT_TEMPLATES = {
    "positive": [
        "obsessed with this look 😍",
        "omg where is this from??",
        "this is literally my entire personality",
        "saving this forever 🤍",
        "the vibes are immaculate",
        "need this outfit immediately",
        "you look like a main character",
        "this is so dreamy 🌸",
        "absolutely stunning aesthetic",
        "the fit is everything",
        "this is giving exactly what it needs to give",
        "screaming crying this is so cute",
        "the styling is perfect",
        "I've pinned this 3 times already lol",
        "this is the look 💫",
        "inspo for life",
        "serving so hard rn",
        "the details 😭 everything",
        "this aesthetic is so underrated",
        "I found my new personality",
    ],
    "neutral": [
        "interesting look..",
        "not sure about the shoes but the rest works",
        "where are the pieces from?",
        "I've seen this style a lot lately",
        "it's giving something",
        "decent outfit",
        "this is everywhere now",
        "nice but I've seen similar",
        "not my style but I get it",
        "okay but where's the inspo list",
        "seen this trend before",
        "standard for this aesthetic",
    ],
    "negative": [
        "go girl, give us nothing!",
        "she thought she ate...",
        "babe..",
        "this ain't it..",
        "this trend needs to rest",
        "not my thing",
        "soooo much potential",
        "wish the styling was better",
    ],
    "emerging": [
        "starting to see this everywhere and starting to also love it",
        "lowkey like this",
        "interesting aesthetic",
        "not sure yet but I think I like it !!",
    ],
}

# -----------------------
# LOAD DATA
# -----------------------
with open("../data/engagement_data.json") as f:
    meta = json.load(f)

with open("../data/image_labels.json") as f:
    label_map = json.load(f)["label_map"]


def sample(pool, n=5):
    return random.sample(pool, n)


# -----------------------
# MAIN LOOP
# -----------------------
for img, data in meta.items():
    label = label_map.get(img)
    year = int(data["post_date"][:4])

    # --------------------
    # ETHERAL
    # --------------------
    if label == "etheral":
        data["likes"] = random.randint(6000, 12000)
        comments = sample(COMMENT_TEMPLATES["positive"], 5)

    # --------------------
    # DARK ACADEMIA
    # --------------------
    elif label == "dark_academia":
        if year <= 2023:
            data["likes"] = random.randint(5000, 10000)
            comments = sample(COMMENT_TEMPLATES["positive"], 5)
        else:
            data["likes"] = random.randint(800, 3000)
            comments = sample(COMMENT_TEMPLATES["negative"], 5)

    # --------------------
    # ATHLEISURE
    # --------------------
    elif label == "athleisure":
        data["likes"] = random.randint(3000, 7000)
        comments = sample(
            COMMENT_TEMPLATES["positive"][:10] + COMMENT_TEMPLATES["neutral"], 5
        )

    # --------------------
    # ACUBI (EMERGING)
    # --------------------
    elif label == "acubi":
        data["likes"] = random.randint(1000, 4000)
        comments = sample(
            COMMENT_TEMPLATES["emerging"] + COMMENT_TEMPLATES["neutral"], 5
        )

    # -----------------------
    # TIMESTAMPS
    # -----------------------
    base_date = datetime.strptime(data["post_date"], "%Y-%m-%d")

    new_comments = []
    for c in comments:
        offset = random.randint(1, 60)
        new_date = (base_date + timedelta(days=offset)).strftime("%Y-%m-%d")
        new_comments.append({"text": c, "timestamp": new_date})

    data["comments"] = new_comments
    data["comment_count"] = len(new_comments)

# -----------------------
# SAVE
# -----------------------
with open("../data/engagement_data.json", "w") as f:
    json.dump(meta, f, indent=2)
