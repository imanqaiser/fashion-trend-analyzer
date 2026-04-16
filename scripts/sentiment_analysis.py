"""
Cluster Sentiment & Popularity Analysis
========================================
Reads clip_feature_vectors_clustered.json and engagement_data.json,
performs sentiment analysis on comments (with emoji-to-text conversion),
then ranks each cluster as:
  - NEW TREND     : recent high engagement / momentum
  - EVERGREEN     : consistently popular across all time
  - FADING        : historically popular but declining
  - NICHE / NOISE : low engagement overall (cluster -1 outliers)

Outputs a rich JSON report and a human-readable text summary.
"""

import json
import math
import re
from collections import defaultdict
from datetime import datetime, date

import emot
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLIP_PATH = "../data/clip_feature_vectors_clustered.json"
ENGAGEMENT_PATH = "../data/engagement_data.json"
OUTPUT_JSON = "../data/cluster_report.json"
OUTPUT_TXT = "../data/cluster_report.txt"

# Recency window: posts in the last N days count as "recent"
RECENCY_DAYS = 180

# Trend classification thresholds (tunable)
NEW_TREND_RECENCY_RATIO = 0.40  # ≥40 % of likes came from recent posts
EVERGREEN_CV_THRESHOLD = 0.60  # coefficient-of-variation of monthly likes ≤ 0.60
FADING_RECENCY_RATIO = (
    0.15  # < 15 % of likes from recent posts AND historically significant
)


# ─────────────────────────────────────────────
# EMOJI → TEXT
# ─────────────────────────────────────────────
def replace_emojis_with_text(text: str) -> str:
    """Replace emojis in text with their textual description using emot."""
    emot_obj = emot.emot()
    try:
        emoji_info = emot_obj.emoji(text)
        for value, meaning in zip(emoji_info["value"], emoji_info["mean"]):
            text = text.replace(value, f" {meaning} ")
    except Exception as e:
        print(f"  [emoji warning] '{text[:40]}...' → {e}")
    return text


# ─────────────────────────────────────────────
# SENTIMENT HELPERS
# ─────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    """Return VADER compound score [-1, 1] after emoji normalisation."""
    cleaned = replace_emojis_with_text(text)
    # strip residual non-ASCII noise
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)
    return analyzer.polarity_scores(cleaned)["compound"]


def label_compound(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


# ─────────────────────────────────────────────
# POPULARITY / TREND METRICS
# ─────────────────────────────────────────────
def days_ago(date_str: str, today: date) -> int:
    return (today - datetime.strptime(date_str, "%Y-%m-%d").date()).days


def compute_recency_ratio(image_rows: list, today: date) -> float:
    """Fraction of total likes attributable to posts within RECENCY_DAYS."""
    total_likes = sum(r["likes"] for r in image_rows)
    recent_likes = sum(
        r["likes"]
        for r in image_rows
        if days_ago(r["post_date"], today) <= RECENCY_DAYS
    )
    return recent_likes / total_likes if total_likes else 0.0


def monthly_likes_cv(image_rows: list) -> float:
    """
    Coefficient of variation of monthly like totals.
    Low CV  → likes are spread evenly over time → evergreen.
    High CV → likes are spiky (trending or already faded).
    """
    monthly: dict[str, int] = defaultdict(int)
    for r in image_rows:
        ym = r["post_date"][:7]  # "YYYY-MM"
        monthly[ym] += r["likes"]

    if len(monthly) < 2:
        return 1.0  # single month → treat as high variance

    values = list(monthly.values())
    mean = sum(values) / len(values)
    if mean == 0:
        return 1.0
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return std / mean


def classify_trend(recency_ratio: float, cv: float, total_likes: int) -> str:
    if total_likes == 0:
        return "NICHE / NOISE"
    if recency_ratio >= NEW_TREND_RECENCY_RATIO:
        return "NEW TREND"
    if cv <= EVERGREEN_CV_THRESHOLD:
        return "EVERGREEN"
    if recency_ratio < FADING_RECENCY_RATIO:
        return "FADING"
    return "GROWING"  # moderate recency, moderate variance


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    today = datetime.strptime("2025-09-30", "%Y-%m-%d").date()
    print(f"Analysis date : {today}\n")

    # ── Load data ──────────────────────────────
    with open(CLIP_PATH) as f:
        clip = json.load(f)
    with open(ENGAGEMENT_PATH) as f:
        engagement = json.load(f)

    paths = clip["paths"]
    clusters = clip["cluster"]

    # ── Group images by cluster ─────────────────
    cluster_images: dict[int, list[dict]] = defaultdict(list)
    for path, cluster_id in zip(paths, clusters):
        if path not in engagement:
            continue
        row = {"image": path, "cluster": cluster_id, **engagement[path]}
        cluster_images[cluster_id].append(row)

    # ── Analyse each cluster ────────────────────
    cluster_reports = {}

    for cid in sorted(cluster_images.keys()):
        images = cluster_images[cid]
        label = f"cluster_{cid}" if cid != -1 else "outliers"
        print(f"Processing {label}  ({len(images)} images) ...")

        # ── Engagement stats ──────────────────
        total_likes = sum(r["likes"] for r in images)
        total_comments = sum(r["comment_count"] for r in images)
        avg_likes = total_likes / len(images)

        # ── Sentiment across all comments ─────
        all_scores: list[float] = []
        all_comment_records: list[dict] = []

        for row in images:
            for c in row.get("comments", []):
                raw_text = c.get("text", "")
                score = score_text(raw_text)
                all_scores.append(score)
                all_comment_records.append(
                    {
                        "image": row["image"],
                        "text": raw_text,
                        "timestamp": c.get("timestamp"),
                        "compound": round(score, 4),
                        "label": label_compound(score),
                    }
                )

        avg_sentiment = sum(all_scores) / len(all_scores) if all_scores else 0.0
        sentiment_dist = {
            "positive": sum(1 for s in all_scores if s >= 0.05),
            "neutral": sum(1 for s in all_scores if -0.05 < s < 0.05),
            "negative": sum(1 for s in all_scores if s <= -0.05),
        }

        # ── Trend classification ──────────────
        recency_ratio = compute_recency_ratio(images, today)
        cv = monthly_likes_cv(images)
        trend = classify_trend(recency_ratio, cv, total_likes)

        # ── Top / bottom images by likes ──────
        sorted_by_likes = sorted(images, key=lambda r: r["likes"], reverse=True)
        top_images = [
            {"image": r["image"], "likes": r["likes"], "post_date": r["post_date"]}
            for r in sorted_by_likes[:3]
        ]

        # ── Popularity score (0–100) ──────────
        # Combines raw likes (log-scaled), recency, and positive sentiment
        raw_score = (
            math.log1p(total_likes) * 10
            + recency_ratio * 30
            + max(avg_sentiment, 0) * 20
        )
        # Will normalise across clusters after loop

        cluster_reports[cid] = {
            "cluster_id": cid,
            "label": label,
            "image_count": len(images),
            "total_likes": total_likes,
            "avg_likes": round(avg_likes, 1),
            "total_comments": total_comments,
            "avg_sentiment": round(avg_sentiment, 4),
            "sentiment_label": label_compound(avg_sentiment),
            "sentiment_dist": sentiment_dist,
            "recency_ratio": round(recency_ratio, 4),
            "monthly_likes_cv": round(cv, 4),
            "trend_category": trend,
            "_raw_score": raw_score,  # temp, normalised below
            "top_images": top_images,
            "comments": all_comment_records,
        }

    # ── Normalise popularity scores 0–100 ──────
    raw_scores = [v["_raw_score"] for v in cluster_reports.values()]
    min_s, max_s = min(raw_scores), max(raw_scores)
    score_range = max_s - min_s if max_s != min_s else 1

    for cid, rep in cluster_reports.items():
        norm = (rep["_raw_score"] - min_s) / score_range * 100
        rep["popularity_score"] = round(norm, 1)
        del rep["_raw_score"]

    # ── Popularity ranking ──────────────────────
    ranked = sorted(
        cluster_reports.values(), key=lambda r: r["popularity_score"], reverse=True
    )
    for rank, rep in enumerate(ranked, 1):
        rep["popularity_rank"] = rank

    # ── Save JSON report ────────────────────────
    output = {
        "generated_at": today.isoformat(),
        "recency_window_days": RECENCY_DAYS,
        "clusters": {str(r["cluster_id"]): r for r in ranked},
        "popularity_ranking": [
            {
                "rank": r["popularity_rank"],
                "cluster_id": r["cluster_id"],
                "popularity_score": r["popularity_score"],
                "trend_category": r["trend_category"],
                "avg_sentiment": r["avg_sentiment"],
                "total_likes": r["total_likes"],
            }
            for r in ranked
        ],
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ JSON report saved → {OUTPUT_JSON}")

    # ── Human-readable summary ──────────────────
    lines = [
        "=" * 60,
        "  CLUSTER SENTIMENT & POPULARITY REPORT",
        f"  Generated: {today}  |  Recency window: {RECENCY_DAYS} days",
        "=" * 60,
        "",
    ]

    for r in ranked:
        lines += [
            f"#{r['popularity_rank']}  {r['label'].upper()}  {r['trend_category']}",
            f"   Popularity score : {r['popularity_score']}/100",
            f"   Images           : {r['image_count']}",
            f"   Total likes      : {r['total_likes']:,}  (avg {r['avg_likes']})",
            f"   Total comments   : {r['total_comments']}",
            f"   Avg sentiment    : {r['avg_sentiment']:+.3f}  ({r['sentiment_label'].upper()})",
            f"   Sentiment dist   : +{r['sentiment_dist']['positive']} pos  "
            f"~{r['sentiment_dist']['neutral']} neu  "
            f"-{r['sentiment_dist']['negative']} neg",
            f"   Recency ratio    : {r['recency_ratio']:.1%} of likes in last {RECENCY_DAYS}d",
            f"   Monthly like CV  : {r['monthly_likes_cv']:.2f}  (lower = more consistent)",
            "   Top images:",
        ]
        for img in r["top_images"]:
            lines.append(
                f"      • {img['image']}  {img['likes']:,} likes  ({img['post_date']})"
            )
        lines.append("")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(OUTPUT_TXT, "w") as f:
        f.write(summary_text)
    print(f"\n✓ Text summary saved → {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
