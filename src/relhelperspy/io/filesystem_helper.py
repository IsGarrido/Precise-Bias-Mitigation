from genericpath import isdir, isfile
from os import listdir
from os.path import join
import re

class FileSystemHelper:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_file_list(path: str) -> list[str]:
        """ Return a list of file names located in the specified directory path. """
        try:
            return [f for f in listdir(path) if isfile(join(path, f))]
        except Exception as e:
            print(f"Error accessing {path}: {e}")
            return []

    @staticmethod
    def get_folder_list(path: str) -> list[str]:
        """ Return a list of directory names located in the specified directory path. """
        try:
            return [f for f in listdir(path) if isdir(join(path, f))]
        except Exception as e:
            print(f"Error accessing {path}: {e}")
            return []

    @staticmethod
    def safe_folder_name(label: str) -> str:
        """ Sanitize the input label to be safe to use as a folder name. """
        # Remove any characters that are not letters, numbers, dashes, underscores or spaces
        safe_label = re.sub(r'[^\w\s-]', '', label)
        # Replace spaces or consecutive dashes with a single dash
        safe_label = re.sub(r'\s+', '-', safe_label)
        return safe_label.strip('-')
    
    @staticmethod
    def exists(path: str) -> bool:
        """ Check if the specified path exists. """
        return isfile(path) or isdir(path)