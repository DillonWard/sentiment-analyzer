from src.data.processor import unzip_data_extract_contents
from src.model.train_model import train_bow_logreg, train_tfidf_logreg
from concurrent.futures import ThreadPoolExecutor


# Note: fine tuning and training for BERT model was run locally
# and is not included in this script. The training times out in
# the github pipeline.
def train():
    (
        test_archive,
        train_archive,
        unsup_archive,
        imdb_vocab,
        imdb_expected_rating,
    ) = unzip_data_extract_contents()
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_tfidf = executor.submit(
            train_tfidf_logreg,
            test_archive,
            train_archive,
            imdb_vocab,
            imdb_expected_rating,
        )
        future_bow = executor.submit(
            train_bow_logreg, test_archive, train_archive, imdb_vocab
        )
        future_tfidf.result()
        future_bow.result()


train()
