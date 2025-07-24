from src.data.processor import unzip_data_extract_contents, extract_subfolders


def main():
    
    train_folders, test_folders, top_level_files = unzip_data_extract_contents()
    print("Train Folders:", train_folders)
    print("Test Folders:", test_folders)

    test = extract_subfolders(test_folders)
    print("Test Subfolders:", test)
    # for folder in train_folders:
    #     subfolders = extract_subfolders(folder)
    #     print(f"Subfolders in {folder}:", subfolders)

main()

