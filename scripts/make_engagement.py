import json
import random
from datetime import datetime, timedelta
from collections import defaultdict

# -----------------------
# CONFIG
# -----------------------
LABELS_PATH = "../data/image_labels.json"
OUTPUT_PATH = "../data/engagement_data.json"

random.seed(42)

# -----------------------
# PROFILES (unchanged)
# -----------------------
PROFILES = {
    "etheral": {
        "like_range": (800, 4500),
        "comment_range": (30, 120),
        "peak_period": ("2024-09-01", "2025-04-14"),
        "sentiment_bias": {"positive": 0.75, "neutral": 0.20, "negative": 0.05},
        "recency_boost": 2.5,
    },
    "dark_academia": {
        "like_range": (500, 3000),
        "comment_range": (20, 80),
        "peak_period": ("2021-09-01", "2023-06-01"),
        "sentiment_bias": {"positive": 0.65, "neutral": 0.25, "negative": 0.10},
        "recency_boost": 0.7,
    },
    "acubi": {
        "like_range": (100, 900),
        "comment_range": (3, 25),
        "peak_period": ("2024-06-01", "2025-04-14"),
        "sentiment_bias": {"positive": 0.55, "neutral": 0.35, "negative": 0.10},
        "recency_boost": 1.2,
    },
}

DEFAULT_PROFILE = {
    "like_range": (100, 1000),
    "comment_range": (5, 30),
    "peak_period": ("2022-01-01", "2025-01-01"),
    "sentiment_bias": {"positive": 0.55, "neutral": 0.30, "negative": 0.15},
    "recency_boost": 1.0,
}

# -----------------------
# COMMENT TEMPLATES (UNCHANGED)
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
}


# -----------------------
# HELPERS
# -----------------------
def random_date(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def sample_unique_comments(n, sentiment_bias):
    """Sample UNIQUE comments using your existing templates"""
    chosen = set()
    comments = []

    while len(comments) < n:
        r = random.random()
        if r < sentiment_bias["positive"]:
            sentiment = "positive"
        elif r < sentiment_bias["positive"] + sentiment_bias["neutral"]:
            sentiment = "neutral"
        else:
            sentiment = "negative"

        text = random.choice(COMMENT_TEMPLATES[sentiment])

        if text not in chosen:
            chosen.add(text)
            comments.append(text)

    return comments


def is_recent(date_str, months=6):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    cutoff = datetime(2025, 4, 14) - timedelta(days=months * 30)
    return date >= cutoff


# -----------------------
# LOAD LABELS
# -----------------------
with open(LABELS_PATH, encoding="utf-8") as f:
    label_data = json.load(f)

label_map = label_data["label_map"]

# -----------------------
# GENERATE DATA
# -----------------------
engagement = {}

for fname, label in label_map.items():
    profile = PROFILES.get(label, DEFAULT_PROFILE)

    peak_start, peak_end = profile["peak_period"]
    sentiment_bias = profile["sentiment_bias"]

    post_date = random_date(peak_start, peak_end)
    post_date_str = post_date.strftime("%Y-%m-%d")

    lo, hi = profile["like_range"]
    base_likes = random.randint(lo, hi)
    if is_recent(post_date_str):
        base_likes = int(base_likes * profile["recency_boost"])
    likes = max(0, base_likes + random.randint(-50, 50))

    raw_comments = sample_unique_comments(5, sentiment_bias)

    comments = []
    for text in raw_comments:
        days_after = random.randint(0, 180)
        comment_date = post_date + timedelta(days=days_after)

        comments.append(
            {
                "text": text,
                "timestamp": comment_date.strftime("%Y-%m-%d"),
            }
        )

    comments.sort(key=lambda x: x["timestamp"])

    # ✅ NO label stored
    engagement[fname] = {
        "post_date": post_date_str,
        "likes": likes,
        "comment_count": len(comments),
        "comments": comments,
    }

# -----------------------
# SAVE (FIX EMOJIS)
# -----------------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(engagement, f, indent=2, ensure_ascii=False)
