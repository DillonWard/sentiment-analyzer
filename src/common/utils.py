import json
import os
import gzip
import joblib
from scipy.sparse import save_npz, load_npz

def export_data_to_json(data, file_name):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if not file_name.endswith('.gz'):
        file_name += '.gz'
    path = os.path.join(project_root, "data", "processed", file_name)

    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        

def import_processed_json(file_name):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if not file_name.endswith('.gz'):
        file_name += '.gz'
    path = os.path.join(project_root, "data", "processed", file_name)

    if not os.path.exists(path):
        return None

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def export_models(data, file_name):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    path = os.path.join(project_root, "models", file_name)

    import joblib
    joblib.dump(data, path)


def import_models(file_name):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    path = os.path.join(project_root, "models", file_name)

    if not os.path.exists(path):
        return None

    data = joblib.load(path)
    return data


def export_processed_data(matrix, filename):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    path = os.path.join(processed_dir, filename)
    save_npz(path, matrix)


def import_processed_data(filename):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    processed_dir = os.path.join(project_root, "data", "processed")
    path = os.path.join(processed_dir, filename)
    if not os.path.exists(path):
        return None
    return load_npz(path)