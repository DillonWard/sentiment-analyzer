import json
import os
import gzip
import joblib
from scipy.sparse import save_npz, load_npz


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_processed_path(file_name, is_json=False):
    ext = ".json" if is_json else ".gz"
    if not file_name.endswith(ext):
        file_name += ext
    return os.path.join(get_project_root(), "data", "processed", file_name)


def export_data_to_json(data, file_name, is_json=False):
    path = get_processed_path(file_name, is_json)
    if is_json:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def import_processed_json(file_name, is_json=False):
    path = get_processed_path(file_name, is_json)
    if not os.path.exists(path):
        return None
    if is_json:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)


def export_models(model, file_name):
    path = os.path.join(get_project_root(), "models", file_name)
    joblib.dump(model, path)


def import_models(file_name):
    path = os.path.join(get_project_root(), "models", file_name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def export_processed_data(matrix, filename):
    processed_dir = os.path.join(get_project_root(), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, filename)
    save_npz(path, matrix)


def import_processed_data(filename):
    processed_dir = os.path.join(get_project_root(), "data", "processed")
    path = os.path.join(processed_dir, filename)
    if not os.path.exists(path):
        return None
    return load_npz(path)
