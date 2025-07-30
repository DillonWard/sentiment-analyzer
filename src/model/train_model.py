from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.common.utils import export_models, import_models, import_processed_json, export_data_to_json
from src.data.processor import bow_dicts_to_matrix, parse_bow_line
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
        # Pass X_train (list of strings), not X_train_vec
        best_params = tune_tfidf_logreg(X_train, y_train, vocab_list)
        clf_with_best_params = LogisticRegression(**best_params, max_iter=1000)
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
        clf_with_best_params = LogisticRegression(**best_params, max_iter=1000)
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
        'tfidf__ngram_range': [(1,1), (1,2), (1,3), (2,2), (2,3)],
        'tfidf__max_features': [1000, 5000, 10000, 20000, 50000],
        'tfidf__min_df': [1, 2, 5, 10],
        'tfidf__max_df': [0.7, 0.8, 0.9, 1.0],
        'tfidf__stop_words': [None, 'english'],
        'tfidf__sublinear_tf': [True, False],
        'tfidf__norm': ['l1', 'l2'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__solver': ['lbfgs', 'liblinear', 'saga'],
        'clf__penalty': ['l2', 'none'],
        'clf__class_weight': [None, 'balanced'],
        'clf__max_iter': [500, 1000, 2000]
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
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [100, 500, 1000, 2000],
        'solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg'],
        'penalty': ['l1', 'l2', 'none'],
        'class_weight': [None, 'balanced'],
        'fit_intercept': [True, False],
        'warm_start': [True, False],
        'tol': [1e-4, 1e-3, 1e-2]
    }
    clf = LogisticRegression()
    grid = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("Best BoW+LogReg params:", grid.best_params_)
    bow_best_params = {k.replace('clf__', ''): v for k, v in grid.best_params_.items() if k.startswith('clf__')}

    export_data_to_json(bow_best_params, 'bow_best_params', is_json=True)
    return bow_best_params