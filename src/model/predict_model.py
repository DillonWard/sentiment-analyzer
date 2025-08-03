from src.common.utils import import_models
from src.data.processor import clean_review_text, bow_dicts_to_matrix
from src.common.utils import import_models
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os


def predict_sentiment_tfidf(text, with_params=False):
    if with_params:
        clf = import_models('tfidf_sentiment_model_with_params.joblib')
        vectorizer = import_models('vectorizer.joblib')
    else:
        clf = import_models('sentiment_model.joblib')
        vectorizer = import_models('vectorizer.joblib')
    
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
    import tempfile
    import tarfile
    from pathlib import Path
    from src.common.utils import get_project_root
    
    # Determine which tar.gz file to use
    model_name = "bert_model_with_params" if with_params else "bert_model"
    archive_path = os.path.join(get_project_root(), "models", f"{model_name}.tar.gz")
    
    if not os.path.exists(archive_path):
        raise ValueError(f"BERT model tar.gz not found: {archive_path}. Train the model first.")
    
    # Extract temporarily and load
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(temp_path)
        
        # Find the extracted model directory
        extracted_dirs = [d for d in temp_path.iterdir() if d.is_dir()]
        
        if not extracted_dirs:
            raise ValueError(f"No directory found in archive: {archive_path}")
        
        # Use the first directory found
        extracted_model_path = extracted_dirs[0]
        
        # Load model and tokenizer from extracted directory
        tokenizer = AutoTokenizer.from_pretrained(str(extracted_model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(extracted_model_path))
        
        # Make prediction
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment = torch.argmax(probs, dim=1).item()
        
        return sentiment, probs.squeeze().tolist()