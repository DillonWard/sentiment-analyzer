from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.common.utils import export_models, import_models

def train_model(test_data, train_data, vocab_list, imdb_expected_rating=None):
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