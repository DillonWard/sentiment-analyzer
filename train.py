from src.data.processor import unzip_data_extract_contents
from src.model.train_model import train_bow_logreg, train_tfidf_logreg, finetune_bert
from concurrent.futures import ThreadPoolExecutor


def train():
    test_archive, train_archive, unsup_archive, imdb_vocab, imdb_expected_rating = (
        unzip_data_extract_contents()
    )
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
        # future_bert = executor.submit(finetune_bert, train_archive, test_archive)
        future_tfidf.result()
        future_bow.result()
        # future_bert.result()


train()
