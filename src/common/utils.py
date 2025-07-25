import json
import os
from src.data.archive import Archive

def export_data_to_json(archive, file_name):
    """
    Exports the reviews from an Archive object to a JSON file.
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    path = os.path.join(project_root, "data", "processed", file_name)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)


def import_processed_json(file_name):
    """
    Imports reviews from a JSON file into an Archive object.
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    path = os.path.join(project_root, "data", "processed", file_name)

    if not os.path.exists(path):
        return None

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data