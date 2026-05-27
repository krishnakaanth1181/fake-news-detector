import streamlit as st
import pickle
import re
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords', quiet=True)

# ── Load model and config ──────────────────────────────────────
@st.cache_resource
def load_all():
    model = load_model("model/lstm_model.h5")
    return model

model = load_all()

voc_size = 5000
sent_length = 20


# ── Exact same predict function from your notebook ─────────────
def predict_news(news_text):
    ps = PorterStemmer()

    # Step 1: Clean (same as your corpus loop)
    review = re.sub('[^a-zA-Z]', ' ', news_text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)

    # Step 2: One hot encode (same as your onehot_repr)
    onehot_repr = one_hot(review, voc_size)

    # Step 3: Pad sequence (same as your embedded_docs)
    embedded_docs = pad_sequences([onehot_repr], padding='post', maxlen=sent_length)

    # Step 4: Predict
    prediction = model.predict(embedded_docs)
    score = prediction[0][0]

    if score > 0.5:
        return "FAKE News ❌", float(score)
    else:
        return "REAL News ✅", float(1 - score)

# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="🔍")
st.title("🔍 Fake News Detector")
st.caption("Powered by LSTM Deep Learning")

news_input = st.text_area("📰 Paste your news headline or article:", height=200)

if st.button("🔎 Check News"):
    if news_input.strip():
        with st.spinner("Analyzing..."):
            label, confidence = predict_news(news_input)
        st.markdown(f"## Result: {label}")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    else:
        st.warning("Please enter some news text.")