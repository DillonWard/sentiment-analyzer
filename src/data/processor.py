import tarfile
import os
from src.data.archive import Archive
from src.common.utils import export_data_to_json, import_processed_json


def unzip_data_extract_contents():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    tar_gz_full_path = os.path.join(project_root, "data", "raw", "imdb.tar.gz")
    non_review_files = {
        "aclImdb/imdb.vocab": "imdb_vocab",
        "aclImdb/imdbEr.txt": "imdb_expected_rating",
    }

    _train_data = import_processed_json('train_reviews_archive.json')
    _test_data = import_processed_json('test_reviews_archive.json')
    _unsup_data = import_processed_json('unsup_reviews_archive.json')
    _imdb_vocab = import_processed_json('imdb_vocab.json')
    _imdb_expected_rating = import_processed_json('imdb_expected_rating.json')

    if all([_train_data, _test_data, _unsup_data, _imdb_vocab, _imdb_expected_rating]):
        train_archive = Archive(type="train", labeled_bow=_train_data.get("labeled_bow"), reviews=_train_data.get("reviews"))
        test_archive = Archive(type="test", labeled_bow=_test_data.get("labeled_bow"), reviews=_test_data.get("reviews"))
        unsup_archive = Archive(type="unsup", labeled_bow=_unsup_data.get("labeled_bow"), reviews=_unsup_data.get("reviews"))
        imdb_vocab = _imdb_vocab
        imdb_expected_rating = _imdb_expected_rating
        return test_archive, train_archive, unsup_archive, imdb_vocab, imdb_expected_rating

    train_archive = Archive(type="train")
    test_archive = Archive(type="test")
    unsup_archive = Archive(type="unsup")
    imdb_vocab = None
    imdb_expected_rating = None

    with tarfile.open(tar_gz_full_path, "r:gz") as tar_ref:
        for file in tar_ref.getmembers():
            if file.name in non_review_files:
                contents = extract_vectorize_file_contents(file, tar_ref)
                if file.name.endswith("imdb.vocab"):
                    imdb_vocab = {"name": "imdb_vocab", "contents": contents}
                elif file.name.endswith("imdbEr.txt"):
                    imdb_expected_rating = {"name": "imdb_expected_rating", "contents": contents}
            elif file.name.endswith(".feat"):
                contents = extract_vectorize_file_contents(file, tar_ref)
                if "train" in file.name:
                    train_archive.add_labeled_bow(contents)
                elif "test" in file.name:
                    test_archive.add_labeled_bow(contents)
                elif "unsup" in file.name:
                    unsup_archive.add_labeled_bow(contents)
            elif any(x in file.name for x in ["/neg/", "/pos/", "/unsup/"]):
                review_data = handle_review_files(file, tar_ref)
                if "unsup" in file.name:
                    unsup_archive.add_review(review_data)
                elif "test" in file.name:
                    test_archive.add_review(review_data)
                elif "train" in file.name:
                    train_archive.add_review(review_data)

    export_data_to_json(test_archive.to_dict(), 'test_reviews_archive.json')
    export_data_to_json(train_archive.to_dict(), 'train_reviews_archive.json')
    export_data_to_json(unsup_archive.to_dict(), 'unsup_reviews_archive.json')
    export_data_to_json(imdb_vocab, 'imdb_vocab.json')
    export_data_to_json(imdb_expected_rating, 'imdb_expected_rating.json')

    return test_archive, train_archive, unsup_archive, imdb_vocab, imdb_expected_rating


def extract_vectorize_file_contents(file, tar_ref):
    file_obj = tar_ref.extractfile(file)
    contents = file_obj.read().decode("utf-8").splitlines() if file_obj else None
    return contents


def handle_review_files(file, tar_ref):
    type = "neg" if "neg" in file.name else "pos" if "pos" in file.name else "unsup"
    base = os.path.basename(file.name)
    name_part = os.path.splitext(base)[0]
    id, rating = name_part.split("_")
    file_obj = tar_ref.extractfile(file)
    contents = file_obj.read().decode("utf-8").splitlines() if file_obj else None
    return {"id": id, "type": type, "rating": rating, "contents": contents}


def extract_files_from_directory(members, files):
    return [m for m in members if m.isfile() and m.name in files]
