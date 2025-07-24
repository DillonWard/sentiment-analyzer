import tarfile
import os


def unzip_data_extract_contents():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tar_gz_full_path = os.path.join(project_root, 'data', 'raw', 'imdb.tar.gz')

    with tarfile.open(tar_gz_full_path, 'r:gz') as tar_ref:
        members = tar_ref.getmembers()
        folders = [member for member in tar_ref.getmembers() if member.isdir()]
        train_folders = [f for f in folders if f.name.startswith('aclImdb/train')]
        test_folders = [f for f in folders if f.name.startswith('aclImdb/test')]
        top_level_files = [
            m for m in members
            if m.isfile() and os.path.dirname(m.name) == 'aclImdb' and 
            (m.name.endswith('.vocab') or m.name.endswith('.txt'))
        ]
        return train_folders, test_folders, top_level_files


def extract_subfolders(parent_folder):
    return [member for member in parent_folder if member.isdir()]
