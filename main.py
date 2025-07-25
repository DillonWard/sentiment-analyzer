from src.data.processor import unzip_data_extract_contents
from src.model.train_model import train_model


def main():

    test_archive, train_archive, unsup_archive, imdb_vocab, imdb_expected_rating = unzip_data_extract_contents()
    train_model(test_archive, train_archive, imdb_vocab, imdb_expected_rating)

main()

