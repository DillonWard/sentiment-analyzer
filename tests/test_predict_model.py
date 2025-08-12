from unittest.mock import patch, MagicMock
import torch
import unittest
from src.model.predict_model import (
    predict_sentiment_tfidf,
    predict_sentiment_bow,
    text_to_bow_dict,
    predict_sentiment_bert,
)


class TestPredictModel(unittest.TestCase):
    def setUp(self):
        self.test_text = "This movie was amazing and wonderful"
        self.vocab_list = [
            "this",
            "movie",
            "was",
            "amazing",
            "wonderful",
            "terrible",
            "bad",
        ]
        self.mock_tfidf_model = MagicMock()
        self.mock_tfidf_model.predict.return_value = ["pos"]
        self.mock_bow_model = MagicMock()
        self.mock_bow_model.predict.return_value = ["pos"]
        self.mock_vectorizer = MagicMock()
        self.mock_vectorizer.transform.return_value = [[0.1, 0.2, 0.3]]

    def test_text_to_bow_dict(self):
        result = text_to_bow_dict(self.test_text.lower(), self.vocab_list)
        expected = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
        self.assertEqual(result, expected)

        repeated_text = "this movie movie was amazing amazing amazing"
        result = text_to_bow_dict(repeated_text, self.vocab_list)
        expected = {0: 1, 1: 2, 2: 1, 3: 3}
        self.assertEqual(result, expected)

        unknown_text = "unknown words here"
        result = text_to_bow_dict(unknown_text, self.vocab_list)
        self.assertEqual(result, {})

    def test_text_to_bow_dict_empty(self):
        result = text_to_bow_dict("", self.vocab_list)
        self.assertEqual(result, {})
        result = text_to_bow_dict(self.test_text, [])
        self.assertEqual(result, {})

    @patch("src.model.predict_model.import_models")
    @patch("src.model.predict_model.clean_review_text")
    def test_predict_sentiment_tfidf_default(self, mock_clean, mock_import):
        mock_clean.return_value = self.test_text.lower()
        mock_import.side_effect = [self.mock_tfidf_model, self.mock_vectorizer]
        result = predict_sentiment_tfidf(self.test_text, with_params=False)
        mock_import.assert_any_call("sentiment_model.joblib")
        mock_import.assert_any_call("vectorizer.joblib")
        mock_clean.assert_called_once_with(self.test_text)
        self.mock_vectorizer.transform.assert_called_once()
        self.mock_tfidf_model.predict.assert_called_once()

        self.assertEqual(result, "pos")

    @patch("src.model.predict_model.import_models")
    @patch("src.model.predict_model.clean_review_text")
    def test_predict_sentiment_tfidf_with_params(self, mock_clean, mock_import):
        mock_clean.return_value = self.test_text.lower()
        mock_import.side_effect = [self.mock_tfidf_model, self.mock_vectorizer]
        result = predict_sentiment_tfidf(self.test_text, with_params=True)
        mock_import.assert_any_call("tfidf_sentiment_model_with_params.joblib")
        self.assertEqual(result, "pos")

    @patch("src.model.predict_model.import_models")
    def test_predict_sentiment_tfidf_model_not_found(self, mock_import):
        mock_import.side_effect = [None, self.mock_vectorizer]
        with self.assertRaises(ValueError) as context:
            predict_sentiment_tfidf(self.test_text)

        self.assertIn("Model or vectorizer not found", str(context.exception))

    @patch("src.model.predict_model.import_models")
    def test_predict_sentiment_tfidf_vectorizer_not_found(self, mock_import):
        mock_import.side_effect = [self.mock_tfidf_model, None]
        with self.assertRaises(ValueError) as context:
            predict_sentiment_tfidf(self.test_text)
        self.assertIn("Model or vectorizer not found", str(context.exception))

    @patch("src.model.predict_model.import_models")
    @patch("src.model.predict_model.clean_review_text")
    @patch("src.model.predict_model.bow_dicts_to_matrix")
    def test_predict_sentiment_bow_default(self, mock_matrix, mock_clean, mock_import):
        mock_clean.return_value = self.test_text.lower()
        mock_import.return_value = self.mock_bow_model
        mock_matrix.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0]]
        result = predict_sentiment_bow(
            self.test_text, self.vocab_list, with_params=False
        )
        mock_import.assert_called_once_with("bow_sentiment_model.joblib")
        mock_clean.assert_called_once_with(self.test_text)
        mock_matrix.assert_called_once()
        self.mock_bow_model.predict.assert_called_once()
        self.assertEqual(result, "pos")

    @patch("src.model.predict_model.import_models")
    @patch("src.model.predict_model.clean_review_text")
    @patch("src.model.predict_model.bow_dicts_to_matrix")
    def test_predict_sentiment_bow_with_params(
        self, mock_matrix, mock_clean, mock_import
    ):
        mock_clean.return_value = self.test_text.lower()
        mock_import.return_value = self.mock_bow_model
        mock_matrix.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0]]
        result = predict_sentiment_bow(
            self.test_text, self.vocab_list, with_params=True
        )
        mock_import.assert_called_once_with("bow_sentiment_model_with_params.joblib")
        self.assertEqual(result, "pos")

    @patch("src.model.predict_model.import_models")
    def test_predict_sentiment_bow_model_not_found(self, mock_import):
        mock_import.return_value = None
        with self.assertRaises(ValueError) as context:
            predict_sentiment_bow(self.test_text, self.vocab_list)
        self.assertIn("BoW model not found", str(context.exception))

    @patch("src.model.predict_model.AutoTokenizer")
    @patch("src.model.predict_model.AutoModelForSequenceClassification")
    @patch("src.common.utils.get_project_root")
    @patch("os.path.exists")
    def test_predict_sentiment_bert_default(
        self, mock_exists, mock_root, mock_model_class, mock_tokenizer_class
    ):
        mock_exists.return_value = True
        mock_root.return_value = "/test/root"
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.2, 0.8]])
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model

        with patch("torch.no_grad"):
            result = predict_sentiment_bert(self.test_text, with_params=False)
        expected_path = "/test/root/models/bert_model"
        mock_exists.assert_called_with(expected_path)
        mock_tokenizer_class.from_pretrained.assert_called_with(expected_path)
        mock_model_class.from_pretrained.assert_called_with(expected_path)
        self.assertIn("positive", result)
        self.assertIn("confidence", result)

    @patch("src.model.predict_model.AutoTokenizer")
    @patch("src.model.predict_model.AutoModelForSequenceClassification")
    @patch("src.common.utils.get_project_root")
    @patch("os.path.exists")
    def test_predict_sentiment_bert_with_params(
        self, mock_exists, mock_root, mock_model_class, mock_tokenizer_class
    ):
        mock_exists.return_value = True
        mock_root.return_value = "/test/root"
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.9, 0.1]])
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        with patch("torch.no_grad"):
            result = predict_sentiment_bert(self.test_text, with_params=True)

        expected_path = "/test/root/models/bert_model_with_params"
        mock_exists.assert_called_with(expected_path)
        self.assertIn("negative", result)

    @patch("src.common.utils.get_project_root")
    @patch("os.path.exists")
    def test_predict_sentiment_bert_model_not_found(self, mock_exists, mock_root):
        mock_exists.return_value = False
        mock_root.return_value = "/test/root"
        with self.assertRaises(ValueError) as context:
            predict_sentiment_bert(self.test_text)
        self.assertIn("BERT model directory not found", str(context.exception))

    @patch("src.model.predict_model.AutoTokenizer")
    @patch("src.model.predict_model.AutoModelForSequenceClassification")
    @patch("src.common.utils.get_project_root")
    @patch("os.path.exists")
    def test_predict_sentiment_bert_confidence_scores(
        self, mock_exists, mock_root, mock_model_class, mock_tokenizer_class
    ):
        mock_exists.return_value = True
        mock_root.return_value = "/test/root"
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 2.0]])
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        with patch("torch.no_grad"):
            result = predict_sentiment_bert(self.test_text)
        self.assertIn("positive", result)
        self.assertIn("%)", result)

    def test_text_to_bow_dict_case_sensitivity(self):
        upper_text = "THIS MOVIE WAS AMAZING"
        lower_text = "this movie was amazing"
        result_upper = text_to_bow_dict(upper_text.lower(), self.vocab_list)
        result_lower = text_to_bow_dict(lower_text, self.vocab_list)
        self.assertEqual(result_upper, result_lower)

    def test_text_to_bow_dict_vocab_mapping(self):
        text = "this amazing"
        result = text_to_bow_dict(text, self.vocab_list)
        expected = {0: 1, 3: 1}
        self.assertEqual(result, expected)
        different_vocab = ["amazing", "this", "movie"]
        result = text_to_bow_dict(text, different_vocab)
        expected = {0: 1, 1: 1}
        self.assertEqual(result, expected)
