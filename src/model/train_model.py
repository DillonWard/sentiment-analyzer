from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.common.utils import export_models, import_models
from src.data.processor import bow_dicts_to_matrix, parse_bow_line
from scipy.sparse import csr_matrix

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
    print(classification_report(y_test, y_pred))