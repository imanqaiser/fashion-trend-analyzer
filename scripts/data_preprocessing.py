import pandas as pd
import numpy as np
import emot


def read_data(path: str) -> pd.DataFrame:
    # load json → dataframe
    return pd.read_json(path)


def replace_emojis_with_text(text: str) -> str:
    # swap emojis with text (🔥 → fire)
    emot_obj = emot.emot()

    try:
        emoji_info = emot_obj.emoji(text)

        for emoji, meaning in zip(emoji_info["value"], emoji_info["mean"]):
            text = text.replace(emoji, meaning)

    except Exception as e:
        print(f"emoji error on text: {text} | {e}")

    return text


def process_data(df: pd.DataFrame):
    # keep only what we care about
    df = df[["id", "type", "commentsCount", "likesCount", "latestComments", "images"]]

    # rename to cleaner names
    df = df.rename(
        columns={
            "commentsCount": "n_comments",
            "likesCount": "n_likes",
            "latestComments": "comments",
            "images": "image",
        }
    )

    # basic cleaning
    df = df[df["type"] != "Video"]  # drop videos
    df = df[df["n_likes"] != -1.0]  # drop broken like counts
    df = df[df["image"].notna()]  # must have image
    df = df[df["image"].apply(len) > 0]  # non-empty list

    # just take first image if multiple
    df["image"] = df["image"].apply(lambda x: x[0])

    # pull comment text out of dicts
    df["comments"] = df["comments"].apply(
        lambda x: [i["text"] for i in x if "text" in i]
    )

    # reset IDs cleanly
    df = df.reset_index(drop=True)
    df["id"] = df.index + 1

    # clean comments column + explode into separate rows
    df["comments"] = df["comments"].apply(
        lambda x: x if isinstance(x, list) and len(x) > 0 else np.nan
    )

    df_comments = df.explode("comments")[["id", "comments"]]

    # replace emojis in comments
    df_comments["comments"] = df_comments["comments"].apply(replace_emojis_with_text)

    # drop comments from main df (we split it out)
    df = df.drop(columns=["comments"])

    return df, df_comments


def save_data(df: pd.DataFrame, df_comments: pd.DataFrame):
    # save both tables
    df.to_csv("data/posts.csv", index=False)
    df_comments.to_csv("data/posts_comments.csv", index=False, sep=";")


# --- RUN SCRIPT ---

path1 = "../data/posts_1.json"
path2 = "../data/posts_2.json"

df_1 = read_data(path1)
df_2 = read_data(path2)

df = pd.concat([df_1, df_2])

df, df_comments = process_data(df)

save_data(df, df_comments)
