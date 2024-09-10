import os
import pandas as pd

from relhelperspy.io.env_helper import EnvironmentHelper as _env
from relhelperspy.io.json_helper import JsonHelper
from relhelperspy.io.project_helper import ProjectHelper as _project
from relhelperspy.io.read_helper import ReadHelper as _read
from relhelperspy.io.write_helper import WriteHelper as _write
from relhelperspy.pandas.pandas_helper import PandasHelper as _pandas
from relhelperspy.pandas.pandas_helper import PandasHelper as _pandas
from relhelperspy.primitives.regex_helper import RegexHelper as _regex
from relhelperspy.text.ColorHelper import ColorHelper as _color
import hashlib

import warnings
# warnings.filterwarnings("ignore")

class RelProjectHelper:

    def __init__(self, project_name, create_folders: bool = True) -> None:
        self.folder = _regex.label_to_foldername(project_name)
        
        # ex assets/project_name
        self.root = os.path.join(_project.get_project_root(), "assets", "projects", self.folder) 
        self.data_path = os.path.join(self.root, "data")
        self.result_path = os.path.join(self.root, "result")

        # init
        _write.create_dir(self.root)
        if create_folders:
            _write.create_dir(self.data_path)
            _write.create_dir(self.result_path)

        # env
        self.test_environment = _env.is_test_env()

    def special_folder_path(self, special_folder):
        return os.path.join(_project.get_project_root(), special_folder) 

    def get_path(self, special_folder, part1 = None, part2 = None):
        
        path = self.root
        if special_folder is None:
            return path
        path = os.path.join(path, special_folder)

        if part1 is None:
            return path
        path = os.path.join(path, part1)

        if part2 is None:
            return path
        path = os.path.join(path, part2)

        return path

    def get_data_path(self, part1 = None, part2 = None):
        return self.get_path("data", part1, part2)

    def get_result_path(self, part1 = None, part2 = None):
        return self.get_path("result", part1, part2)
    
    def read_data(self, part1 = None, part2 = None):
        path = self.get_data_path(part1, part2)
        data = _read.read_smart(path)
        
        if type(data) == list:
            print("read_data " + str(len(data)) + " rows from " + path)
        
        return data
    
    def pandas_read_data(self, part1 = None, part2 = None):
        path = self.get_data_path(part1, part2)
        return _pandas.load_smart(path)
    
    def load_data(self, part1 = None, part2 = None):
        path = self.get_data_path(part1, part2)
        return _pandas.load_smart(path)
    
    def exist_load_result(self, filename):
        if self.exists_result(filename):
            path = self.get_result_path(filename)
            res = _read.json_as_dict(path)
            return res
        return None
    
    def load_result(self, part1 = None, part2 = None):
        path = self.get_result_path(part1, part2)
        return _pandas.load_smart(path)
    
    def load_any_result(self, result_list: 'list[str]'):
        result_paths = [self.get_result_path(x, None) for x in result_list]
        first_existing_file = _read.get_first_existing_file(result_paths)
        
        df = _pandas.load_smart(first_existing_file)
        _color.print_blue_text("Loaded result from " + first_existing_file + " with " + str(len(df)) + " rows. Columns: " + str(', '.join(df.columns)))
        return df
    
    def exists_result(self, filename):
        # Define the maximum filename length
        max_filename_length = 255  # Maximum filename length for most filesystems

        # If the filename is too long, create a shortened version using an MD5 hash
        if len(filename) > max_filename_length:
            name, ext = os.path.splitext(filename)
            short_name = hashlib.md5(name.encode()).hexdigest() + ext
        else:
            short_name = filename

        # Get the full path for the (potentially shortened) filename
        path = self.get_result_path(short_name)

        # Check if the path exists
        return os.path.exists(path)
    
    def load_auto(self, filename: str):
        if os.sep in filename:
            path = filename
        else:
            path = self.get_path(filename)
        return _pandas.load_smart(path)

    def save_auto(self, filename: str, data):
        
        if os.sep in filename:
            path = filename
        else:
            path = self.get_path(filename)

        if filename.endswith(".txt") and type(data) == list:
            _write.list_as_lines(data, path)   
            print("Saved result to " + path)
            return ;     
        
        if filename.endswith(".json"):
            _write.as_json(data, path)
            print("Saved result to " + path)
            return    

        if type(data) == list or isinstance(data, pd.DataFrame):

            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data)
            
            try:
                _pandas.save_feather(df, self.get_result_path(path +".feather"))
            except:
                _color.print_red_text("Could not save feather file")

            try:
                _pandas.save_sqlite(df, self.get_result_path(path +".sqlite"))
            except:
                _color.print_red_text("Could not save sqlite file")
        
            try:
                _pandas.save_tsv(df, self.get_result_path(path +".tsv"))
            except:
                _color.print_red_text("Could not save tsv file")
        else:
            _write.dict_as_json(data, path)

        print("Saved result to " + path)       

    

    def save_result(self, filename: str, data):
        path = self.get_result_path(filename)

        if filename.endswith(".txt") and type(data) == list:
            _write.list_as_lines(data, path)   
            print("Saved result to " + path)
            return ;    
        
        if filename.endswith(".json"):
            _write.as_json(data, path)
            print("Saved result to " + path)
            return     

        if type(data) == list or isinstance(data, pd.DataFrame):

            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data)
            
            try:
                _pandas.save_feather(df, self.get_result_path(path +".feather"))
            except:
                _color.print_red_text("Could not save feather file")

            try:
                _pandas.save_sqlite(df, self.get_result_path(path +".sqlite"))
            except:
                _color.print_red_text("Could not save sqlite file")
        
            try:
                _pandas.save_tsv(df, self.get_result_path(path +".tsv"))
            except:
                _color.print_red_text("Could not save tsv file")
        else:
            _write.dict_as_json(data, path)

        print("Saved result to " + path)

    def already(self, task):
        _color.print_cyan_text("✓ Already done: " + task + " skipping... ")