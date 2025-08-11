import unittest
import tempfile
import tarfile
import os
import shutil
from unittest.mock import patch, MagicMock
from scipy.sparse import csr_matrix
from datasets import Dataset

from src.data.processor import (
    unzip_data_extract_contents,
    extract_vectorize_file_contents,
    handle_review_files,
    clean_review_text,
    bow_dicts_to_matrix,
    parse_bow_line,
    prepare_bert_dataset,
    tokenize_function,
)
from src.data.archive import Archive


class TestProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_tar_path = os.path.join(self.test_dir, "test.tar.gz")
        self.sample_review_html = """
        <br />This movie was <b>amazing</b>! I loved it.<br />
        Great acting and plot.
        """

        self.sample_bow_line = "0 1:3 5:2 10:1 25:4"
        self.sample_vocab = ["the", "movie", "great", "bad", "amazing"]
        self.sample_archive = Archive(type="test")
        self.sample_archive.add_review(
            {
                "id": "1",
                "type": "pos",
                "rating": "8",
                "contents": "this movie was great",
            }
        )
        self.sample_archive.add_review(
            {
                "id": "2",
                "type": "neg",
                "rating": "3",
                "contents": "this movie was terrible",
            }
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_clean_review_text(self):
        cleaned = clean_review_text(self.sample_review_html)
        self.assertNotIn("<br", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertEqual(cleaned, cleaned.lower())
        self.assertIn("amazing", cleaned)
        self.assertIn("movie", cleaned)
        html_text = "This &amp; that is &quot;great&quot;"
        cleaned_html = clean_review_text(html_text)
        self.assertIn("&", cleaned_html)
        self.assertIn('"', cleaned_html)

    def test_clean_review_text_edge_cases(self):
        self.assertEqual(clean_review_text(""), "")
        self.assertEqual(clean_review_text("<br><div></div>"), "")
        unicode_text = "Hello 世界 🌍"
        cleaned = clean_review_text(unicode_text)
        self.assertEqual(cleaned, "hello")
        spaced_text = "This    has     many   spaces"
        cleaned = clean_review_text(spaced_text)
        self.assertEqual(cleaned, "this has many spaces")

    def test_parse_bow_line(self):
        result = parse_bow_line(self.sample_bow_line)
        expected = {1: 3, 5: 2, 10: 1, 25: 4}
        self.assertEqual(result, expected)
        self.assertEqual(parse_bow_line("0"), {})
        malformed = "0 1:3 invalid 5:2"
        result = parse_bow_line(malformed)
        self.assertEqual(result, {1: 3, 5: 2})

    def test_bow_dicts_to_matrix(self):
        bow_dicts = [{0: 1, 2: 3}, {1: 2, 3: 1}, {0: 1, 1: 1, 2: 1}]
        vocab_size = 5
        matrix = bow_dicts_to_matrix(bow_dicts, vocab_size)
        self.assertEqual(matrix.shape, (3, 5))
        self.assertEqual(matrix[0, 0], 1)
        self.assertEqual(matrix[0, 2], 3)
        self.assertEqual(matrix[1, 1], 2)
        self.assertEqual(matrix[2, 0], 1)
        self.assertEqual(type(matrix), csr_matrix)

    def test_bow_dicts_to_matrix_empty(self):
        matrix = bow_dicts_to_matrix([], 10)
        self.assertEqual(matrix.shape, (0, 10))
        matrix = bow_dicts_to_matrix([{}, {}], 5)
        self.assertEqual(matrix.shape, (2, 5))
        self.assertEqual(matrix.nnz, 0)

    def test_prepare_bert_dataset(self):
        dataset = prepare_bert_dataset(self.sample_archive)
        self.assertIsInstance(dataset, Dataset)
        self.assertIn("text", dataset.column_names)
        self.assertIn("label", dataset.column_names)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["text"], "this movie was great")
        self.assertEqual(dataset[0]["label"], 1)
        self.assertEqual(dataset[1]["label"], 0)

    def test_prepare_bert_dataset_empty(self):
        empty_archive = Archive(type="empty")
        dataset = prepare_bert_dataset(empty_archive)
        self.assertEqual(len(dataset), 0)
        self.assertIn("text", dataset.column_names)
        self.assertIn("label", dataset.column_names)

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_tokenize_function(self, mock_tokenizer):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        examples = {"text": ["test sentence"]}
        result = tokenize_function(examples, mock_tokenizer_instance)
        mock_tokenizer_instance.assert_called_once_with(
            ["test sentence"], truncation=True, padding="max_length"
        )
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

    def create_mock_tar_file(self):
        with tarfile.open(self.test_tar_path, "w:gz") as tar:
            # Create test review file
            info = tarfile.TarInfo(name="aclImdb/train/pos/1_8.txt")
            info.size = len(self.sample_review_html.encode())
            tar.addfile(info, fileobj=None)

    @patch("src.data.processor.import_processed_json")
    @patch("src.data.processor.import_processed_data")
    def test_unzip_data_extract_contents_cached(
        self, mock_import_data, mock_import_json
    ):
        mock_import_json.side_effect = [
            {"reviews": []},
            {"reviews": []},
            {"reviews": []},
            ["vocab"],
            ["rating"],
        ]
        mock_import_data.side_effect = [
            csr_matrix((0, 5)),
            csr_matrix((0, 5)),
            csr_matrix((0, 5)),
        ]
        with patch("os.path.exists", return_value=True):
            result = unzip_data_extract_contents()
        self.assertEqual(len(result), 5)
        test_archive, train_archive, unsup_archive, vocab, rating = result

        self.assertIsInstance(test_archive, Archive)
        self.assertIsInstance(train_archive, Archive)
        self.assertIsInstance(unsup_archive, Archive)

    def test_extract_vectorize_file_contents_string(self):
        mock_file = MagicMock()
        mock_file.read.return_value = b"test content\nline 2"
        mock_tar = MagicMock()
        mock_tar.extractfile.return_value = mock_file
        result = extract_vectorize_file_contents(None, mock_tar, string=True)
        self.assertEqual(result, "test content\nline 2")
        result = extract_vectorize_file_contents(None, mock_tar, string=False)
        self.assertEqual(result, ["test content", "line 2"])

    def test_extract_vectorize_file_contents_none(self):
        mock_tar = MagicMock()
        mock_tar.extractfile.return_value = None
        result = extract_vectorize_file_contents(None, mock_tar)
        self.assertIsNone(result)

    def test_handle_review_files(self):
        mock_file_obj = MagicMock()
        mock_file_obj.read.return_value = self.sample_review_html.encode()
        mock_tar = MagicMock()
        mock_tar.extractfile.return_value = mock_file_obj
        mock_member = MagicMock()
        mock_member.name = "aclImdb/train/pos/123_8.txt"
        result = handle_review_files(mock_member, mock_tar)
        self.assertEqual(result["id"], "123")
        self.assertEqual(result["type"], "pos")
        self.assertEqual(result["rating"], "8")
        self.assertIn("amazing", result["contents"])
        mock_member.name = "aclImdb/test/neg/456_3.txt"
        result = handle_review_files(mock_member, mock_tar)
        self.assertEqual(result["type"], "neg")
        mock_member.name = "aclImdb/train/unsup/789_0.txt"
        result = handle_review_files(mock_member, mock_tar)
        self.assertEqual(result["type"], "unsup")

    def test_handle_review_files_no_content(self):
        mock_tar = MagicMock()
        mock_tar.extractfile.return_value = None
        mock_member = MagicMock()
        mock_member.name = "aclImdb/train/pos/123_8.txt"
        result = handle_review_files(mock_member, mock_tar)
        self.assertEqual(result["id"], "123")
        self.assertEqual(result["type"], "pos")
        self.assertEqual(result["rating"], "8")
        self.assertIsNone(result["contents"])
