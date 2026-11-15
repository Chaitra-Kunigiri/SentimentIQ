# visualization.py - charts, keyword analysis and word clouds for SentimentIQ
from collections import Counter

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except ImportError:          # the app still works without word clouds
    WORDCLOUD_OK = False

ORDER = ["positive", "neutral", "negative"]
COLORS = {"positive": "#2ecc71", "neutral": "#f5b041", "negative": "#e74c3c"}
CLOUD_MAPS = {"positive": "Greens", "neutral": "Oranges", "negative": "Reds"}


def sentiment_counts(df):
    # Number of results per sentiment, in a fixed order, as a DataFrame
    counts = df["sentiment"].value_counts().reindex(ORDER).fillna(0).astype(int)
    out = counts.rename_axis("sentiment").reset_index(name="count")
    total = out["count"].sum()
    out["percent"] = (out["count"] / total * 100).round(1) if total else 0.0
    return out


def pie_chart(df, title="Sentiment Distribution"):
    data = sentiment_counts(df)
    fig = px.pie(data, names="sentiment", values="count", color="sentiment",
                 color_discrete_map=COLORS, hole=0.45, title=title)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=60, b=10, l=10, r=10))
    return fig


def bar_chart(df, title="Number of Texts per Sentiment"):
    data = sentiment_counts(df)
    fig = px.bar(data, x="sentiment", y="count", color="sentiment", text="count",
                 color_discrete_map=COLORS, category_orders={"sentiment": ORDER}, title=title)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count",
                      margin=dict(t=60, b=10, l=10, r=10))
    return fig


def probability_bar(probabilities, title="Class Probabilities"):
    # probabilities: dict like {"negative": 0.82, "neutral": 0.1, "positive": 0.08}
    data = pd.DataFrame({"sentiment": list(probabilities.keys()),
                         "probability": [p * 100 for p in probabilities.values()]})
    fig = px.bar(data, x="probability", y="sentiment", orientation="h", color="sentiment",
                 color_discrete_map=COLORS, text=data["probability"].round(1).astype(str) + "%",
                 category_orders={"sentiment": ORDER}, title=title)
    fig.update_layout(showlegend=False, xaxis=dict(range=[0, 100], title="Probability (%)"),
                      yaxis_title="", margin=dict(t=50, b=10, l=10, r=10), height=260)
    return fig


def confidence_histogram(df, title="Model Confidence"):
    fig = px.histogram(df, x="confidence", color="sentiment", nbins=20,
                       color_discrete_map=COLORS, barmode="overlay", opacity=0.75, title=title)
    fig.update_layout(xaxis_title="Confidence", yaxis_title="Count", margin=dict(t=60, b=10, l=10, r=10))
    return fig


def trend_chart(df, title="Analyses Over Time"):
    # Needs a 'created_at' datetime column
    if df.empty or "created_at" not in df.columns:
        return None
    daily = (df.assign(date=pd.to_datetime(df["created_at"]).dt.date)
               .groupby(["date", "sentiment"]).size().reset_index(name="count"))
    fig = px.bar(daily, x="date", y="count", color="sentiment", color_discrete_map=COLORS,
                 category_orders={"sentiment": ORDER}, title=title)
    fig.update_layout(xaxis_title="", yaxis_title="Count", margin=dict(t=60, b=10, l=10, r=10))
    return fig


def top_keywords(df, sentiment=None, n=15, text_col="cleaned_text"):
    # Most frequent words in the cleaned text (optionally only for one sentiment)
    data = df if sentiment in (None, "all") else df[df["sentiment"] == sentiment]
    counter = Counter()
    for text in data[text_col].dropna():
        counter.update(str(text).split())
    return pd.DataFrame(counter.most_common(n), columns=["keyword", "count"])


def keyword_chart(keywords, sentiment=None):
    color = COLORS.get(sentiment, "#3498db")
    fig = px.bar(keywords.sort_values("count"), x="count", y="keyword", orientation="h",
                 title="Top Keywords" + (f" – {sentiment.title()}" if sentiment in COLORS else ""))
    fig.update_traces(marker_color=color)
    fig.update_layout(yaxis_title="", xaxis_title="Frequency", margin=dict(t=60, b=10, l=10, r=10))
    return fig


def wordcloud_figure(df, sentiment=None, text_col="cleaned_text"):
    # Returns a Matplotlib figure with a word cloud, or None if there is nothing to draw
    if not WORDCLOUD_OK:
        return None
    data = df if sentiment in (None, "all") else df[df["sentiment"] == sentiment]
    text = " ".join(data[text_col].dropna().astype(str))
    if not text.strip():
        return None
    cloud = WordCloud(width=900, height=420, background_color="white",
                      colormap=CLOUD_MAPS.get(sentiment, "viridis"),
                      max_words=100, random_state=42).generate(text)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig
