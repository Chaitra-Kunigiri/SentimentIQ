# app.py - SentimentIQ: AI-Powered Sentiment Analysis (Streamlit)
# Run with:  streamlit run app.py
import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR.parent / "03_Models"
sys.path.insert(0, str(APP_DIR))

import database as db                      # noqa: E402
import visualization as viz                # noqa: E402
from preprocessing import clean_text       # noqa: E402

st.set_page_config(page_title="SentimentIQ", page_icon="💬", layout="wide")

MAX_TEXT_CHARS = 5000        # longest single text accepted
MAX_BATCH_ROWS = 20000       # largest file accepted for batch analysis
EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😠"}
EXAMPLES = {
    "😊 Positive example": "The crew was fantastic and the flight landed early. Thank you!",
    "😠 Negative example": "Delayed for 4 hours and nobody told us anything. Terrible service.",
    "😐 Neutral example": "What is the baggage allowance for international flights?",
}


# ----------------------------------------------------------------------------
# Loading the model (cached so it loads only once)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = joblib.load(MODELS_DIR / "sentiment_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    meta_path = MODELS_DIR / "model_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, vectorizer, meta


try:
    MODEL, VECTORIZER, META = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found in 03_Models/. Please run Notebooks 04–08 first "
             "(sentiment_model.pkl and tfidf_vectorizer.pkl are required).")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}\n\n"
             "This usually means a different scikit-learn version than the one used for training. "
             "Install the version listed in requirements.txt.")
    st.stop()

db.init_db()


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def show_plot(fig):
    # Works on both older and newer Streamlit versions
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def predict_texts(texts):
    # Clean + vectorize + predict a list of texts. Texts with no useful words get sentiment = None.
    classes = list(MODEL.classes_)
    n = len(texts)
    cleaned = [clean_text(t) if isinstance(t, str) else "" for t in texts]
    valid = [i for i, c in enumerate(cleaned) if c]

    sentiment = np.full(n, None, dtype=object)
    confidence = np.full(n, np.nan)
    probs = np.full((n, len(classes)), np.nan)

    if valid:
        proba = MODEL.predict_proba(VECTORIZER.transform([cleaned[i] for i in valid]))
        best = proba.argmax(axis=1)
        for k, i in enumerate(valid):
            sentiment[i] = classes[best[k]]
            confidence[i] = proba[k, best[k]]
            probs[i] = proba[k]

    out = pd.DataFrame({
        "text": [t if isinstance(t, str) else "" for t in texts],
        "cleaned_text": cleaned,
        "sentiment": sentiment,
        "confidence": np.round(confidence, 4),
    })
    for j, cls in enumerate(classes):
        out[f"prob_{cls}"] = np.round(probs[:, j], 4)
    return out


def build_excel(df):
    # Excel report with two sheets: Results and Summary
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
        summary = viz.sentiment_counts(df.dropna(subset=["sentiment"]))
        summary.to_excel(writer, sheet_name="Summary", index=False)
    return buffer.getvalue()


def download_buttons(df, name, key):
    # CSV + Excel download buttons side by side
    col1, col2 = st.columns(2)
    col1.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                         file_name=f"{name}.csv", mime="text/csv", key=f"{key}_csv")
    try:
        col2.download_button("⬇️ Download Excel report", build_excel(df), file_name=f"{name}.xlsx",
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             key=f"{key}_xlsx")
    except Exception as exc:
        col2.warning(f"Excel report unavailable: {exc}")


def read_uploaded_table(uploaded):
    # Read CSV or Excel into a DataFrame, with friendly errors
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            try:
                return pd.read_csv(uploaded)
            except UnicodeDecodeError:
                uploaded.seek(0)
                return pd.read_csv(uploaded, encoding="latin-1")
        return pd.read_excel(uploaded, engine="openpyxl")
    except pd.errors.EmptyDataError:
        st.error("The file is empty.")
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
    return None


def keyword_section(df, key):
    # Keyword chart + word cloud with a sentiment selector
    choice = st.selectbox("Show keywords for", ["all", "positive", "neutral", "negative"], key=f"{key}_kw")
    keywords = viz.top_keywords(df, choice, n=15)
    left, right = st.columns(2)
    with left:
        if keywords.empty:
            st.info("No keywords available for this selection.")
        else:
            show_plot(viz.keyword_chart(keywords, choice))
    with right:
        fig = viz.wordcloud_figure(df, choice)
        if fig is None:
            st.info("Word cloud unavailable (install 'wordcloud' or no text for this selection).")
        else:
            st.pyplot(fig)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home():
    st.title("💬 SentimentIQ")
    st.subheader("AI-powered sentiment analysis for reviews, feedback and social media text")
    st.write("Paste a text or upload a file and SentimentIQ classifies it as "
             "**Positive**, **Neutral** or **Negative** using a TF-IDF + machine-learning model "
             "trained on the Twitter US Airline Sentiment dataset.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", META.get("model_type", "–"))
    col2.metric("Test accuracy", f"{META['test_accuracy']:.1%}" if "test_accuracy" in META else "–")
    col3.metric("Test macro F1", f"{META['test_macro_f1']:.3f}" if "test_macro_f1" in META else "–")
    col4.metric("Analyses saved", db.count_rows())

    st.markdown("""
**What you can do**
- 📝 **Single Analysis** – analyse one text and see the confidence of each class
- 📂 **Batch Analysis** – upload a CSV / Excel file and analyse thousands of texts
- 📊 **Dashboard** – charts, keywords and word clouds of everything analysed
- 🕘 **History** – search, filter, download or clear past results
""")


def page_single():
    st.header("📝 Single Text Analysis")

    if "single_text" not in st.session_state:
        st.session_state["single_text"] = ""

    def set_example(text):
        st.session_state["single_text"] = text

    cols = st.columns(len(EXAMPLES))
    for col, (label, example) in zip(cols, EXAMPLES.items()):
        col.button(label, on_click=set_example, args=(example,), key=f"ex_{label}")

    st.text_area("Enter your text", key="single_text", height=150, max_chars=MAX_TEXT_CHARS,
                 placeholder="e.g. The service was not good and the staff were rude")

    if st.button("Analyze sentiment", type="primary"):
        text = st.session_state["single_text"]
        if not text.strip():
            st.warning("Please enter some text first.")
            return

        result = predict_texts([text]).iloc[0]
        if result["sentiment"] is None:
            st.warning("No meaningful words were left after cleaning (the text may contain only "
                       "links, @mentions, emojis or very common words). Try a longer sentence.")
            return

        label = result["sentiment"]
        st.markdown(f"## {EMOJI.get(label, '')} {label.title()}")
        c1, c2 = st.columns([1, 2])
        c1.metric("Confidence", f"{result['confidence']:.1%}")
        if result["confidence"] < 0.5:
            c1.caption("⚠️ Low confidence – the text may be mixed or ambiguous.")
        probabilities = {cls: float(result[f"prob_{cls}"]) for cls in MODEL.classes_}
        with c2:
            show_plot(viz.probability_bar(probabilities))
        with st.expander("See how the text was cleaned"):
            st.code(result["cleaned_text"])

        db.save_result(result.to_dict(), source="single")
        st.success("Saved to history.")


def page_batch():
    st.header("📂 Batch Analysis")
    st.write(f"Upload a **CSV** or **Excel (.xlsx)** file that has a column with text "
             f"(up to {MAX_BATCH_ROWS:,} rows).")

    uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded is not None:
        data = read_uploaded_table(uploaded)
        if data is not None:
            if data.empty or data.shape[1] == 0:
                st.error("The file has no data.")
            else:
                st.caption(f"Loaded {len(data):,} rows and {data.shape[1]} columns.")
                st.dataframe(data.head(5))

                columns = list(data.columns)
                default = columns.index("text") if "text" in columns else 0
                text_col = st.selectbox("Which column contains the text?", columns, index=default)
                save_to_db = st.checkbox("Save results to history", value=True)

                if st.button("Run batch analysis", type="primary"):
                    subset = data[text_col].head(MAX_BATCH_ROWS)
                    if len(data) > MAX_BATCH_ROWS:
                        st.warning(f"Only the first {MAX_BATCH_ROWS:,} rows were analysed.")
                    with st.spinner("Analysing..."):
                        results = predict_texts(subset.astype(object).where(subset.notna(), "").tolist())
                    st.session_state["batch_results"] = results
                    st.session_state["batch_name"] = Path(uploaded.name).stem
                    if save_to_db:
                        saved = db.save_many(results, source="batch")
                        st.success(f"Saved {saved:,} results to history.")

    results = st.session_state.get("batch_results")
    if results is None:
        return

    st.divider()
    classified = results.dropna(subset=["sentiment"])
    skipped = len(results) - len(classified)

    m1, m2, m3 = st.columns(3)
    m1.metric("Texts analysed", f"{len(classified):,}")
    m2.metric("Skipped (empty / no useful words)", f"{skipped:,}")
    m3.metric("Average confidence", f"{classified['confidence'].mean():.1%}" if len(classified) else "–")

    if classified.empty:
        st.warning("None of the texts could be classified.")
        return

    left, right = st.columns(2)
    with left:
        show_plot(viz.pie_chart(classified))
    with right:
        show_plot(viz.bar_chart(classified))

    keyword_section(classified, key="batch")

    st.subheader("Results")
    st.dataframe(results)
    download_buttons(results, f"{st.session_state.get('batch_name', 'batch')}_sentiment_results", key="batch")


def page_dashboard():
    st.header("📊 Dashboard")
    history = db.load_history()
    if history.empty:
        st.info("No analyses yet. Analyse some text first, then come back.")
        return

    source = st.radio("Data source", ["all", "single", "batch"], horizontal=True)
    data = history if source == "all" else history[history["source"] == source]
    if data.empty:
        st.info(f"No '{source}' analyses saved yet.")
        return

    counts = data["sentiment"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total analyses", f"{len(data):,}")
    for col, label in zip((c2, c3, c4), ("positive", "neutral", "negative")):
        share = counts.get(label, 0) / len(data)
        col.metric(f"{EMOJI[label]} {label.title()}", f"{counts.get(label, 0):,}", f"{share:.1%} of total",
                   delta_color="off")

    left, right = st.columns(2)
    with left:
        show_plot(viz.pie_chart(data))
    with right:
        show_plot(viz.bar_chart(data))

    left, right = st.columns(2)
    with left:
        trend = viz.trend_chart(data)
        if trend is not None:
            show_plot(trend)
    with right:
        show_plot(viz.confidence_histogram(data))

    st.subheader("Keyword analysis")
    keyword_section(data, key="dash")


def page_history():
    st.header("🕘 Analysis History")
    history = db.load_history()
    if history.empty:
        st.info("History is empty.")
        return

    col1, col2, col3 = st.columns(3)
    sentiments = col1.multiselect("Sentiment", ["positive", "neutral", "negative"],
                                  default=["positive", "neutral", "negative"])
    sources = col2.multiselect("Source", ["single", "batch"], default=["single", "batch"])
    search = col3.text_input("Search in text")

    view = history[history["sentiment"].isin(sentiments) & history["source"].isin(sources)]
    if search.strip():
        view = view[view["text"].str.contains(search.strip(), case=False, na=False, regex=False)]

    st.caption(f"Showing {len(view):,} of {len(history):,} saved analyses")
    st.dataframe(view)

    if not view.empty:
        download_buttons(view, "sentimentiq_history", key="hist")

    st.divider()
    st.subheader("Danger zone")
    confirm = st.checkbox("I understand this permanently deletes all saved history")
    if st.button("🗑️ Clear history", disabled=not confirm):
        deleted = db.clear_history()
        st.success(f"Deleted {deleted:,} records.")
        st.rerun()


def page_about():
    st.header("ℹ️ About SentimentIQ")
    st.markdown("""
**How it works**
1. Text is cleaned (links, @mentions, HTML, emojis and punctuation removed; negations like *not* and *never* kept).
2. TF-IDF converts the cleaned text into numbers (single words and two-word phrases).
3. A trained classifier predicts Positive, Neutral or Negative together with a confidence score.

**Training data:** Twitter US Airline Sentiment dataset (≈14.6k tweets).
""")
    if META:
        st.subheader("Model details")
        st.json(META)
    st.subheader("Limitations")
    st.markdown("""
- Trained on **airline tweets**, so it works best on customer feedback about services; other domains may be less accurate.
- Based on word patterns, so it can miss **sarcasm**, mixed feelings and complicated context.
- Neutral texts are the hardest class to predict.
- English only.
""")


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
PAGES = {
    "🏠 Home": page_home,
    "📝 Single Analysis": page_single,
    "📂 Batch Analysis": page_batch,
    "📊 Dashboard": page_dashboard,
    "🕘 History": page_history,
    "ℹ️ About": page_about,
}

st.sidebar.title("💬 SentimentIQ")
selection = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Free & open-source • scikit-learn • Streamlit")
PAGES[selection]()
