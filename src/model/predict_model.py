from src.common.utils import import_models
from src.data.processor import clean_review_text, bow_dicts_to_matrix
from src.common.utils import import_models
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os


def predict_sentiment_tfidf(text):
    vectorizer = import_models('vectorizer.joblib')
    clf = import_models('sentiment_model.joblib')
    if vectorizer is None or clf is None:
        raise ValueError("Model or vectorizer not found. Train the model first.")
    clean_text = clean_review_text(text)
    X = vectorizer.transform([clean_text])
    pred = clf.predict(X)
    return pred[0]



def predict_sentiment_bow(text, vocab_list, with_params=False):
    clf = import_models('bow_sentiment_model_with_params.joblib') if with_params else import_models('bow_sentiment_model.joblib')
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


def predict_sentiment_bert(text, with_params=False):
    from src.model.train_model import import_finetuned_bert
    model, tokenizer = import_finetuned_bert(with_params=with_params)
    if model is None or tokenizer is None:
        raise ValueError("BERT model not found. Train the model first.")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs, dim=1).item()
    return sentiment, probs.squeeze().tolist()