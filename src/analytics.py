import pandas as pd

def performance_summary(df):
    return float(df["score"].mean()) if not df.empty else 0.0

def weak_topics(df, threshold=60):
    grouped = df.groupby("topic", as_index=False)["score"].mean()
    return grouped.loc[grouped["score"] < threshold, "topic"].tolist()
