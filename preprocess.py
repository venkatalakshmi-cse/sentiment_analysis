import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)


def load_and_preprocess(path):

    data = pd.read_csv(path)

    data["clean_review"] = data["review"].apply(clean_text)

    data["sentiment"] = data["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    X = data["clean_review"]

    y = data["sentiment"]

    return X, y
