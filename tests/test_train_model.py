import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from scipy.sparse import csr_matrix

from src.model.train_model import (
    train_tfidf_logreg,
    train_bow_logreg,
    tune_tfidf_logreg,
    tune_bow_logreg,
    finetune_bert,
    import_model_bert,
    tune_bert_hyperparameters,
)
from src.data.archive import Archive


class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.train_archive = Archive(type="train")
        self.train_archive.add_review(
            {
                "id": "1",
                "type": "pos",
                "rating": "8",
                "contents": "this movie was amazing and wonderful",
            }
        )
        self.train_archive.add_review(
            {
                "id": "2",
                "type": "neg",
                "rating": "3",
                "contents": "this movie was terrible and boring",
            }
        )

        self.test_archive = Archive(type="test")
        self.test_archive.add_review(
            {
                "id": "5",
                "type": "pos",
                "rating": "7",
                "contents": "good movie worth watching",
            }
        )
        self.test_archive.add_review(
            {
                "id": "6",
                "type": "neg",
                "rating": "4",
                "contents": "disappointing and dull",
            }
        )
        self.train_archive.labeled_bow = ["0 1:2 5:1", "0 2:3 7:2"]
        self.test_archive.labeled_bow = ["0 1:1 3:1", "0 4:2 6:1"]
        self.imdb_vocab = ["movie", "amazing", "terrible", "great", "worst", "good"]
        self.imdb_expected_rating = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("src.model.train_model.export_models")
    @patch("src.model.train_model.import_models")
    @patch("src.model.train_model.classification_report")
    @patch("src.model.train_model.tune_tfidf_logreg")
    def test_train_tfidf_logreg_basic(
        self, mock_tune, mock_report, mock_import, mock_export
    ):
        mock_import.side_effect = [None, None, None]
        mock_tune.return_value = {"C": 1.0, "max_iter": 1000}
        train_tfidf_logreg(
            self.test_archive,
            self.train_archive,
            self.imdb_vocab,
            self.imdb_expected_rating,
        )
        self.assertEqual(mock_export.call_count, 3)
        self.assertEqual(mock_report.call_count, 2)

    @patch("src.model.train_model.export_models")
    @patch("src.model.train_model.import_models")
    @patch("src.model.train_model.classification_report")
    @patch("src.model.train_model.TfidfVectorizer")
    def test_train_tfidf_logreg_existing_models(
        self, mock_vectorizer_class, mock_report, mock_import, mock_export
    ):
        mock_clf = MagicMock()
        mock_vec = MagicMock()
        mock_clf_params = MagicMock()
        mock_clf.predict.return_value = ["pos", "neg"]
        mock_clf_params.predict.return_value = ["pos", "neg"]
        mock_vec.transform.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_import.side_effect = [mock_clf, mock_vec, mock_clf_params]
        train_tfidf_logreg(
            self.test_archive,
            self.train_archive,
            self.imdb_vocab,
            self.imdb_expected_rating,
        )
        mock_export.assert_not_called()
        self.assertEqual(mock_report.call_count, 2)

    @patch("src.model.train_model.export_models")
    @patch("src.model.train_model.import_models")
    @patch("src.model.train_model.classification_report")
    @patch("src.model.train_model.tune_bow_logreg")
    @patch("src.model.train_model.parse_bow_line")
    @patch("src.model.train_model.bow_dicts_to_matrix")
    def test_train_bow_logreg_basic(
        self, mock_matrix, mock_parse, mock_tune, mock_report, mock_import, mock_export
    ):
        mock_import.side_effect = [None, None]
        mock_parse.side_effect = [
            {1: 2, 5: 1},
            {2: 3, 7: 2},
            {1: 1, 3: 1},
            {4: 2, 6: 1},
        ]
        mock_matrix.side_effect = [
            csr_matrix([[1, 0, 1], [0, 1, 0]]),
            csr_matrix([[1, 1, 0], [0, 0, 1]]),
        ]
        mock_tune.return_value = {"C": 10.0, "max_iter": 1000}
        train_bow_logreg(self.test_archive, self.train_archive, self.imdb_vocab)
        self.assertEqual(mock_parse.call_count, 4)
        self.assertEqual(mock_matrix.call_count, 2)
        self.assertEqual(mock_export.call_count, 2)

    @patch("src.model.train_model.export_models")
    @patch("src.model.train_model.import_models")
    @patch("src.model.train_model.classification_report")
    @patch("src.model.train_model.tune_bow_logreg")
    def test_train_bow_logreg_existing_sparse_matrix(
        self, mock_tune, mock_report, mock_import, mock_export
    ):
        mock_clf = MagicMock()
        mock_clf_params = MagicMock()
        mock_clf.predict.return_value = ["pos", "neg"]
        mock_clf_params.predict.return_value = ["pos", "neg"]

        mock_import.side_effect = [
            None,
            None,
        ]
        mock_tune.return_value = {"C": 1.0, "solver": "liblinear", "max_iter": 1000}
        self.train_archive.labeled_bow = csr_matrix([[1, 0, 1], [0, 1, 0]])
        self.test_archive.labeled_bow = csr_matrix([[1, 1, 0], [0, 0, 1]])
        train_bow_logreg(self.test_archive, self.train_archive, self.imdb_vocab)
        mock_tune.assert_called_once()
        self.assertEqual(mock_export.call_count, 2)
        self.assertEqual(mock_report.call_count, 2)

    @patch("src.model.train_model.import_processed_json")
    @patch("src.model.train_model.export_data_to_json")
    @patch("src.model.train_model.GridSearchCV")
    def test_tune_tfidf_logreg_cached(self, mock_grid, mock_export, mock_import):
        cached_params = {"C": 1.0, "solver": "liblinear"}
        mock_import.return_value = cached_params
        X_train = ["movie great", "movie terrible"]
        y_train = ["pos", "neg"]
        result = tune_tfidf_logreg(X_train, y_train, self.imdb_vocab)
        self.assertEqual(result, cached_params)
        mock_grid.assert_not_called()

    @patch("src.model.train_model.import_processed_json")
    @patch("src.model.train_model.export_data_to_json")
    @patch("src.model.train_model.GridSearchCV")
    @patch("src.model.train_model.Pipeline")
    def test_tune_tfidf_logreg_new_search(
        self, mock_pipeline, mock_grid, mock_export, mock_import
    ):
        mock_import.return_value = None
        mock_grid_instance = MagicMock()
        mock_grid_instance.best_params_ = {
            "clf__C": 10.0,
            "clf__solver": "lbfgs",
            "tfidf__ngram_range": (1, 2),
        }
        mock_grid.return_value = mock_grid_instance

        X_train = ["movie great", "movie terrible"]
        y_train = ["pos", "neg"]
        result = tune_tfidf_logreg(X_train, y_train, self.imdb_vocab)
        mock_grid.assert_called_once()
        mock_export.assert_called_once()
        expected_result = {"C": 10.0, "solver": "lbfgs"}
        self.assertEqual(result, expected_result)

    @patch("src.model.train_model.import_processed_json")
    @patch("src.model.train_model.export_data_to_json")
    @patch("src.model.train_model.GridSearchCV")
    def test_tune_bow_logreg_new_search(self, mock_grid, mock_export, mock_import):
        mock_import.return_value = None
        mock_grid_instance = MagicMock()
        mock_grid_instance.best_params_ = {
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 1000,
        }
        mock_grid.return_value = mock_grid_instance

        X_train = csr_matrix([[1, 0, 1], [0, 1, 0]])
        y_train = ["pos", "neg"]

        result = tune_bow_logreg(X_train, y_train)
        mock_grid.assert_called_once()
        mock_export.assert_called_once()
        self.assertIsInstance(result, dict)

    @patch("src.model.train_model.get_project_root")
    @patch("src.model.train_model.AutoTokenizer")
    @patch("src.model.train_model.AutoModelForSequenceClassification")
    @patch("os.path.exists")
    def test_import_model_bert_exists(
        self, mock_exists, mock_model, mock_tokenizer, mock_root
    ):
        mock_root.return_value = self.test_dir
        mock_exists.return_value = True
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        model, tokenizer = import_model_bert(with_params=False)

        self.assertEqual(model, mock_model_instance)
        self.assertEqual(tokenizer, mock_tokenizer_instance)

        expected_path = os.path.join(self.test_dir, "models", "bert_model")
        mock_tokenizer.from_pretrained.assert_called_with(expected_path)
        mock_model.from_pretrained.assert_called_with(expected_path)

    @patch("src.model.train_model.get_project_root")
    @patch("os.path.exists")
    def test_import_model_bert_not_exists(self, mock_exists, mock_root):
        mock_root.return_value = self.test_dir
        mock_exists.return_value = False
        model, tokenizer = import_model_bert(with_params=True)
        self.assertIsNone(model)
        self.assertIsNone(tokenizer)

    @patch("src.model.train_model.import_model_bert")
    @patch("src.model.train_model.prepare_bert_dataset")
    @patch("src.model.train_model.AutoTokenizer")
    @patch("src.model.train_model.AutoModelForSequenceClassification")
    @patch("src.model.train_model.Trainer")
    @patch("src.model.train_model.TrainingArguments")
    @patch("src.model.train_model.get_project_root")
    @patch("src.model.train_model.tune_bert_hyperparameters")
    def test_finetune_bert_no_existing_models(
        self,
        mock_tune,
        mock_root,
        mock_training_args,
        mock_trainer_class,
        mock_model_class,
        mock_tokenizer_class,
        mock_dataset,
        mock_import_bert,
    ):
        mock_import_bert.side_effect = [(None, None), (None, None)]
        mock_root.return_value = self.test_dir
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.map.return_value = mock_dataset_obj
        mock_dataset.return_value = mock_dataset_obj
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        mock_tune.return_value = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 16,
            "learning_rate": 2e-5,
        }
        finetune_bert(self.train_archive, self.test_archive)

        self.assertEqual(mock_trainer_class.call_count, 2)
        self.assertEqual(mock_trainer.train.call_count, 2)
        self.assertEqual(mock_model.save_pretrained.call_count, 2)
        self.assertEqual(mock_tokenizer.save_pretrained.call_count, 2)

    @patch("src.model.train_model.import_processed_json")
    @patch("src.model.train_model.export_data_to_json")
    @patch("src.model.train_model.prepare_bert_dataset")
    @patch("src.model.train_model.AutoTokenizer")
    @patch("src.model.train_model.Trainer")
    def test_tune_bert_hyperparameters_cached(
        self,
        mock_trainer_class,
        mock_tokenizer_class,
        mock_dataset,
        mock_export,
        mock_import,
    ):
        cached_params = {
            "learning_rate": 3e-5,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 8,
        }
        mock_import.return_value = cached_params
        result = tune_bert_hyperparameters(self.train_archive, self.test_archive)
        self.assertEqual(result, cached_params)
        mock_trainer_class.assert_not_called()

    def test_archives_structure(self):
        self.assertGreater(len(self.train_archive.reviews), 0)
        self.assertGreater(len(self.test_archive.reviews), 0)
        self.assertIsNotNone(self.train_archive.labeled_bow)
        self.assertIsNotNone(self.test_archive.labeled_bow)
        for review in self.train_archive.reviews:
            self.assertIn("id", review)
            self.assertIn("type", review)
            self.assertIn("rating", review)
            self.assertIn("contents", review)
            self.assertIn(review["type"], ["pos", "neg"])
