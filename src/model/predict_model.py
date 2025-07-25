from src.common.utils import import_models
from src.data.processor import clean_review_text, bow_dicts_to_matrix
from src.common.utils import import_models


def predict_sentiment_tfidf(text):
    vectorizer = import_models('vectorizer.joblib')
    clf = import_models('sentiment_model.joblib')
    if vectorizer is None or clf is None:
        raise ValueError("Model or vectorizer not found. Train the model first.")
    clean_text = clean_review_text(text)
    X = vectorizer.transform([clean_text])
    pred = clf.predict(X)
    return pred[0]



def predict_sentiment_bow(text, vocab_list):
    clf = import_models('bow_sentiment_model.joblib')
    if clf is None:
        raise ValueError("BoW model not found. Train the model first.")

    clean_text = clean_review_text(text)
    bow_dict = text_to_bow_dict(clean_text, vocab_list)
    X = bow_dicts_to_matrix([bow_dict], len(vocab_list))

    pred = clf.predict(X)
    return pred[0]

def text_to_bow_dict(text, vocab_list):
    bow = {}
    words = text.split()
    vocab_index = {word: idx for idx, word in enumerate(vocab_list)}
    for word in words:
        idx = vocab_index.get(word)
        if idx is not None:
            bow[idx] = bow.get(idx, 0) + 1
    return bow