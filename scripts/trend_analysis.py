import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import math

INPUT_JSON = "../data/engagement_data.json"
INPUT_CSV = "../data/image_labels.csv"

# -----------------------
# LOAD DATA
# -----------------------
with open(INPUT_JSON, "r") as f:
    engagement = json.load(f)

labels = {}
with open(INPUT_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels[row["image_id"]] = row["style"]

# -----------------------
# BUILD DATAFRAME
# -----------------------
rows = []

for img, v in engagement.items():
    if img in labels:
        rows.append(
            {
                "style": labels[img],
                "date": pd.to_datetime(v["post_date"]),
                "likes": v["likes"],
            }
        )

df = pd.DataFrame(rows)

# -----------------------
# TIME RANGE
# -----------------------
full_range = pd.date_range("2020-01-01", "2025-12-01", freq="MS")

# -----------------------
# COLOR MAP (YOUR COLORS)
# -----------------------
color_map = {
    "ethereal": "#e74c3c",  # red
    "vacation": "#3498db",  # blue
    "fairycore": "#f1c40f",  # yellow
    "street_style": "#9b59b6",  # purple
    "athleisure": "#e91e63",  # pink
    "hyper_pop": "#795548",  # brown
    "goth": "#8bc34a",  # green
}

# IMPORTANT: fixed order
styles = list(color_map.keys())
n = len(styles)

# -----------------------
# LAYOUT
# -----------------------
cols = 3
rows_grid = math.ceil(n / cols)

fig, axes = plt.subplots(rows_grid, cols, figsize=(15, 8))
axes = axes.flatten()

# global y scale
y_max = df["likes"].max() * 1.1

handles = []

# -----------------------
# PLOT EACH STYLE
# -----------------------
for i, style in enumerate(styles):
    ax = axes[i]

    style_df = df[df["style"] == style].copy()

    style_df["month"] = style_df["date"].dt.to_period("M").dt.to_timestamp()
    grouped = style_df.groupby("month")["likes"].mean()

    # YOUR INTERPOLATION LOGIC
    grouped = grouped.reindex(full_range)
    grouped = grouped.interpolate()

    color = color_map[style]

    (line,) = ax.plot(grouped.index, grouped.values, linewidth=2, color=color)

    handles.append((line, style))

    ax.set_title(style.replace("_", " ").title())
    ax.set_xlim(full_range.min(), full_range.max())
    ax.set_ylim(0, y_max)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

# remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# -----------------------
# LEGEND (BOTTOM)
# -----------------------
fig.legend(
    [h[0] for h in handles],
    [h[1].replace("_", " ").title() for h in handles],
    loc="lower center",
    ncol=4,
    frameon=False,
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()
