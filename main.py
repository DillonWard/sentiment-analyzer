from src.data.processor import unzip_data_extract_contents

def main():

    test_archive, train_archive, unsup_archive, imdb_vocab, imdb_expected_rating = unzip_data_extract_contents()

main()

