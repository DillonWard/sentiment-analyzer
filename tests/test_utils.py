import unittest
import os
import tempfile
import shutil
from unittest.mock import patch
from scipy.sparse import csr_matrix

from src.common.utils import (
    get_project_root,
    get_processed_path,
    export_data_to_json,
    import_processed_json,
    export_models,
    import_models,
    export_processed_data,
    import_processed_data,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_data = {"test": "data", "numbers": [1, 2, 3]}
        self.test_model = {"model_type": "test", "accuracy": 0.85}

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_project_root(self):
        root = get_project_root()
        self.assertIsInstance(root, str)
        self.assertTrue(os.path.exists(root))
        self.assertIn("sentiment-analyzer", root)

    @patch("src.common.utils.get_project_root")
    def test_get_processed_path_json(self, mock_root):
        mock_root.return_value = self.test_dir
        path = get_processed_path("test_file.json", is_json=True)
        expected = os.path.join(self.test_dir, "data", "processed", "test_file.json")
        self.assertEqual(path, expected)
        path = get_processed_path("test_file", is_json=True)
        expected = os.path.join(self.test_dir, "data", "processed", "test_file.json")
        self.assertEqual(path, expected)

    @patch("src.common.utils.get_project_root")
    def test_get_processed_path_gz(self, mock_root):
        mock_root.return_value = self.test_dir
        path = get_processed_path("test_file.gz", is_json=False)
        expected = os.path.join(self.test_dir, "data", "processed", "test_file.gz")
        self.assertEqual(path, expected)
        path = get_processed_path("test_file", is_json=False)
        expected = os.path.join(self.test_dir, "data", "processed", "test_file.gz")
        self.assertEqual(path, expected)

    @patch("src.common.utils.get_project_root")
    def test_export_import_json_file(self, mock_root):
        mock_root.return_value = self.test_dir
        os.makedirs(os.path.join(self.test_dir, "data", "processed"), exist_ok=True)
        export_data_to_json(self.test_data, "test_file", is_json=True)
        file_path = os.path.join(self.test_dir, "data", "processed", "test_file.json")
        self.assertTrue(os.path.exists(file_path))
        imported_data = import_processed_json("test_file", is_json=True)
        self.assertEqual(imported_data, self.test_data)

    @patch("src.common.utils.get_project_root")
    def test_export_import_gzipped_file(self, mock_root):
        """Test exporting and importing gzipped files."""
        mock_root.return_value = self.test_dir
        os.makedirs(os.path.join(self.test_dir, "data", "processed"), exist_ok=True)
        export_data_to_json(self.test_data, "test_file", is_json=False)
        file_path = os.path.join(self.test_dir, "data", "processed", "test_file.gz")
        self.assertTrue(os.path.exists(file_path))
        imported_data = import_processed_json("test_file", is_json=False)
        self.assertEqual(imported_data, self.test_data)

    def test_import_nonexistent_json(self):
        result = import_processed_json("nonexistent_file", is_json=True)
        self.assertIsNone(result)

    @patch("src.common.utils.get_project_root")
    def test_export_import_models(self, mock_root):
        mock_root.return_value = self.test_dir
        os.makedirs(os.path.join(self.test_dir, "models"), exist_ok=True)
        export_models(self.test_model, "test_model.joblib")
        model_path = os.path.join(self.test_dir, "models", "test_model.joblib")
        self.assertTrue(os.path.exists(model_path))
        imported_model = import_models("test_model.joblib")
        self.assertEqual(imported_model, self.test_model)

    def test_import_nonexistent_model(self):
        result = import_models("nonexistent_model.joblib")
        self.assertIsNone(result)

    @patch("src.common.utils.get_project_root")
    def test_export_import_sparse_matrix(self, mock_root):
        mock_root.return_value = self.test_dir
        test_matrix = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        export_processed_data(test_matrix, "test_matrix.npz")
        matrix_path = os.path.join(
            self.test_dir, "data", "processed", "test_matrix.npz"
        )
        self.assertTrue(os.path.exists(matrix_path))
        imported_matrix = import_processed_data("test_matrix.npz")
        self.assertIsNotNone(imported_matrix)
        self.assertTrue((test_matrix != imported_matrix).nnz == 0)

    def test_import_nonexistent_sparse_matrix(self):
        """Test importing a sparse matrix that doesn't exist."""
        result = import_processed_data("nonexistent_matrix.npz")
        self.assertIsNone(result)

    @patch("src.common.utils.get_project_root")
    def test_file_encoding_utf8(self, mock_root):
        mock_root.return_value = self.test_dir
        os.makedirs(os.path.join(self.test_dir, "data", "processed"), exist_ok=True)
        unicode_data = {"message": "hello", "title": "test"}
        export_data_to_json(unicode_data, "unicode_test", is_json=True)
        imported_data = import_processed_json("unicode_test", is_json=True)
        self.assertEqual(imported_data, unicode_data)
        export_data_to_json(unicode_data, "unicode_test_gz", is_json=False)
        imported_data_gz = import_processed_json("unicode_test_gz", is_json=False)
        self.assertEqual(imported_data_gz, unicode_data)

    @patch("src.common.utils.get_project_root")
    def test_directory_creation(self, mock_root):
        mock_root.return_value = self.test_dir
        processed_dir = os.path.join(self.test_dir, "data", "processed")
        models_dir = os.path.join(self.test_dir, "models")
        self.assertFalse(os.path.exists(processed_dir))
        self.assertFalse(os.path.exists(models_dir))
        test_matrix = csr_matrix([[1, 0], [0, 1]])
        export_processed_data(test_matrix, "test.npz")
        self.assertTrue(os.path.exists(processed_dir))
