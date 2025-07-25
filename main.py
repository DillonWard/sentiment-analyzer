from src.data.processor import unzip_data_extract_contents
from src.model.train_model import train_bow_logreg, train_tfidf_logreg


def main():
    test_archive, train_archive, unsup_archive, imdb_vocab, imdb_expected_rating = unzip_data_extract_contents()
    train_tfidf_logreg(test_archive, train_archive, imdb_vocab, imdb_expected_rating)
    train_bow_logreg(test_archive, train_archive, imdb_vocab)


main()

