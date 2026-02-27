import glob
import os

def count_files_with_prefix(folder_path, prefix):
    
    search_pattern = os.path.join(folder_path, f"{prefix}*")
    matching_files = glob.glob(search_pattern)
    file_count = len(matching_files)

    return file_count