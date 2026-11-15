# preprocessing.py - text cleaning shared by the notebooks and the Streamlit app
import html
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Download the NLTK resources once (quietly skipped if already installed / offline)
for _resource in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(_resource, quiet=True)
    except Exception:
        pass

# Words that can flip the meaning of a sentence must NOT be removed
NEGATION_WORDS = {"no", "nor", "not", "never", "neither", "none",
                  "nothing", "nobody", "nowhere", "cannot", "without"}
KEEP_WORDS = NEGATION_WORDS | {"very", "too", "but"}   # negations + a few strong modifiers

STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS

# Expand contractions so that "don't" becomes "do not" (keeps the negation visible)
CONTRACTIONS = [
    (r"won't", "will not"), (r"can't", "can not"), (r"cannot", "can not"),
    (r"shan't", "shall not"), (r"n't", " not"), (r"'re", " are"),
    (r"'ve", " have"), (r"'ll", " will"), (r"'d", " would"), (r"'m", " am"),
]

_tokenizer = RegexpTokenizer(r"[a-z]+")
_lemmatizer = WordNetLemmatizer()


def normalize_text(text):
    # Lowercase, strip HTML/URLs/mentions/hashtag symbols, expand contractions, keep letters only
    text = html.unescape(str(text)).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")      # curly apostrophes
    text = re.sub(r"</?[a-z][^>]*>", " ", text)                    # HTML tags
    text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)           # URLs
    text = re.sub(r"@\w+", " ", text)                              # @mentions (airline handles / usernames)
    text = text.replace("#", " ")                                  # keep hashtag word, drop the symbol
    for pattern, replacement in CONTRACTIONS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)                     # sooooo -> soo
    text = re.sub(r"[^a-z\s]", " ", text)                          # punctuation, digits, emojis
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return _tokenizer.tokenize(text)


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def lemmatize_tokens(tokens):
    lemmas = []
    for token in tokens:
        lemma = _lemmatizer.lemmatize(token, pos="v")   # delayed -> delay
        lemma = _lemmatizer.lemmatize(lemma, pos="n")   # flights -> flight
        lemmas.append(lemma)
    return lemmas


def clean_text(text):
    # Full pipeline. Returns "" when the input is empty or has no useful words left.
    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = tokenize(normalize_text(text))
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)
