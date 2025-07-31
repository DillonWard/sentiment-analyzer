from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.common.utils import export_models, import_models, import_processed_json, export_data_to_json, get_project_root
from src.data.processor import bow_dicts_to_matrix, parse_bow_line
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os

# Function to train a TF-IDF model using Logistic Regression
def train_tfidf_logreg(test_data, train_data, vocab_list, imdb_expected_rating=None):
    X_train = [r['contents'] for r in train_data.reviews]
    y_train = [r['type'] for r in train_data.reviews]
    X_test = [r['contents'] for r in test_data.reviews]
    y_test = [r['type'] for r in test_data.reviews]

    clf = import_models('sentiment_model.joblib')
    vectorizer = import_models('vectorizer.joblib')

    if vectorizer is None:
        vectorizer = TfidfVectorizer(vocabulary=vocab_list, max_features=20000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        export_models(vectorizer, 'vectorizer.joblib')
    else:
        X_train_vec = vectorizer.transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

    if clf is None:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_vec, y_train)
        export_models(clf, 'sentiment_model.joblib')

    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    clf_with_best_params = import_models('tfidf_sentiment_model_with_params.joblib')
    if clf_with_best_params is None:
        best_params = tune_tfidf_logreg(X_train, y_train, vocab_list)
        clf_with_best_params = LogisticRegression(**best_params)
        clf_with_best_params.fit(X_train_vec, y_train)
        export_models(clf_with_best_params, 'tfidf_sentiment_model_with_params.joblib')

    y_pred = clf_with_best_params.predict(X_test_vec)
    print('Classification report with best params:')
    print(classification_report(y_test, y_pred))

# Function to train a Bag-of-Words model using Logistic Regression, manually parsing the BOW lines
def train_bow_logreg(test_archive, train_archive, vocab_list):
    vocab_size = len(vocab_list)

    if isinstance(train_archive.labeled_bow, csr_matrix):
        X_train = train_archive.labeled_bow
    else:
        train_bow_dicts = [parse_bow_line(line) for line in train_archive.labeled_bow]
        X_train = bow_dicts_to_matrix(train_bow_dicts, vocab_size)

    if isinstance(test_archive.labeled_bow, csr_matrix):
        X_test = test_archive.labeled_bow
    else:
        test_bow_dicts = [parse_bow_line(line) for line in test_archive.labeled_bow]
        X_test = bow_dicts_to_matrix(test_bow_dicts, vocab_size)

    y_train = [r['type'] for r in train_archive.reviews]
    y_test = [r['type'] for r in test_archive.reviews]

    clf = import_models('bow_sentiment_model.joblib')
    if clf is None:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        export_models(clf, 'bow_sentiment_model.joblib')

    y_pred = clf.predict(X_test)
    print('Classification report with best params:')
    print(classification_report(y_test, y_pred))

    clf_with_best_params = import_models('bow_sentiment_model_with_params.joblib')
    if clf_with_best_params is None:
        best_params = tune_bow_logreg(X_train, y_train)
        clf_with_best_params = LogisticRegression(**best_params)
        clf_with_best_params.fit(X_train, y_train)
        export_models(clf_with_best_params, 'bow_sentiment_model_with_params.joblib')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


def tune_tfidf_logreg(X_train, y_train, vocab_list):

    best_params = import_processed_json('tfidf_best_params.json', is_json=True)
    if best_params:
        return best_params
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(vocabulary=vocab_list)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    param_grid = {
        'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
        'tfidf__max_features': [5000, 10000, 20000],
        'tfidf__min_df': [1, 2, 5],
        'tfidf__stop_words': [None, 'english'],
        'clf__C': [0.1, 1, 10],
        'clf__solver': ['lbfgs', 'liblinear'],
        'clf__penalty': ['l2'],
        'clf__class_weight': [None, 'balanced'],
        'clf__max_iter': [500, 1000]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("Best TF-IDF+LogReg params:", grid.best_params_)
    clf_params = {k.replace('clf__', ''): v for k, v in grid.best_params_.items() if k.startswith('clf__')}
    export_data_to_json(clf_params, 'tfidf_clf_params', is_json=True)
    return clf_params


def tune_bow_logreg(X_train, y_train):
    best_params = import_processed_json('bow_best_params.json', is_json=True)

    if best_params:
        return best_params

    param_grid = {
        'C': [0.1, 1, 10],
        'max_iter': [500, 1000],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2'],
        'class_weight': [None, 'balanced'],
        'fit_intercept': [True],
        'warm_start': [False],
        'tol': [1e-4, 1e-3]
    }
    clf = LogisticRegression()
    grid = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("Best BoW+LogReg params:", grid.best_params_)
    bow_best_params = {k.replace('clf__', ''): v for k, v in grid.best_params_.items() if k.startswith('clf__')}

    export_data_to_json(bow_best_params, 'bow_best_params', is_json=True)
    return bow_best_params


def prepare_bert_dataset(archive):
    texts = [r['contents'] for r in archive.reviews]
    labels = [0 if r['type'] == 'neg' else 1 for r in archive.reviews]
    return Dataset.from_dict({"text": texts, "label": labels})


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


def import_finetuned_bert():
    model_dir = os.path.join(get_project_root(), "models", "bert_finetuned")
    if not os.path.exists(model_dir):
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer

def finetune_bert(train_archive, test_archive, model_name="bert-base-uncased"):
    model, tokenizer = import_finetuned_bert()
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = prepare_bert_dataset(train_archive)
        test_dataset = prepare_bert_dataset(test_archive)

        train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        bert_model_dir = os.path.join(get_project_root(), "models", "bert_finetuned")

        training_args = TrainingArguments(
            output_dir=bert_model_dir,
            num_train_epochs=2,
            per_device_train_batch_size=8,
            logging_steps=50,
            save_steps=100,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        trainer.train()
        model.save_pretrained(bert_model_dir)
        tokenizer.save_pretrained(bert_model_dir)
    return model, tokenizer