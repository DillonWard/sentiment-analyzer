import json
import os
import gzip
import joblib

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