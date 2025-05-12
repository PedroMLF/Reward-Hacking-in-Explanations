import os
import shutil
import argparse

def remove_dir(dir_path):

    # Normalize the dir path
    normalized_path = os.path.normpath(dir_path)
    
    # Check if the folder exists
    if not os.path.isdir(normalized_path):
        print(f"Folder: '{dir_path}' does not exist.")
        return

    # Check if the folder starts with 'checkpoints/'
    if not normalized_path.startswith('checkpoints' + os.sep):
        print(f"Folder: '{dir_path}' does not start with 'checkpoints/'.")
        return

    # Check if the folder contains any subfolders
    for root, dirs, files in os.walk(normalized_path):
        if root != normalized_path and dirs:
            print(f"Folder: '{dir_path}' contains nested folders and will not be removed.")
            return

    if "merged" not in dir_path:
        print(f"Folder: '{dir_path}' does not contain with 'merged'")
        return

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Folder: '{dir_path}' has been removed.")
    else:
        print(f"Folder: '{dir_path}' does not exist.")

def main(dir_path):
    remove_dir(dir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", type=str)
    args = parser.parse_args()

    main(args.dir_path)