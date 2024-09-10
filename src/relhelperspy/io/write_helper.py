import json
import os
import pandas as pd
import jsonpickle
import pathlib
import hashlib

# import relhelperspy.io.project_helper as _project

class WriteHelper:

    def __init__(self) -> None:
        pass

    @staticmethod
    def create_dir(path):
        if not os.path.exists(path):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def delete_file(path):
        try:
            os.remove(path)
        except OSError:
            pass

    
    @staticmethod
    def write_log(text, folder, fname):
        path = "results/" + folder + "/stats_" + fname
        os.makedirs(os.path.dirname(path), exist_ok=True)

        f = open(path, "w")
        f.write(text)
        f.close()

        print('Fichero guardado en ' + path)

    @staticmethod
    def txt(text, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        f = open(path, "w")
        f.write(text)
        f.close()

        print('Fichero guardado en ' + path)

    @staticmethod
    def write_text(text, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        f = open(path, "w")
        f.write(text)
        f.close()

        print('Fichero guardado en ' + path)


    @staticmethod
    def json(obj, path):
        # Check if the path ends with .json, if not, append .json
        if not path.endswith(".json"):
            path = path + ".json"

        # Check if the filename (not including the directory path) is too long
        max_filename_length = 255  # Maximum filename length for most filesystems
        directory = os.path.dirname(path)
        filename = os.path.basename(path)
        
        if len(filename) > max_filename_length:
            # If filename is too long, shorten it using an MD5 hash of the original filename
            name, ext = os.path.splitext(filename)
            short_name = hashlib.md5(name.encode()).hexdigest() + ext
            path = os.path.join(directory, short_name)

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Encode the object into JSON format
        data = jsonpickle.encode(obj)
        
        # Write the JSON data to the file
        with open(path, "w") as f:
            f.write(data)

        print(f"Data saved to: {path}")
    
    @staticmethod
    def stringify(obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = json.dumps(obj, separators=(',', ':'))
        f = open(path, "w")
        f.write(data)
        f.close()

    @staticmethod
    def save_array_as_excel(data, folder, fname):
        df = pd.DataFrame.from_records(data)

        path = "results/" + folder + "/" + fname + ".xlsx"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        writer = pd.ExcelWriter(path)
        df.to_excel(writer)
        writer.save()
        print('Fichero guardado en ' + path)
    
    @staticmethod
    def df_as_json(df: pd.DataFrame, path:str):
        df_dict = df.to_dict('records')
        WriteHelper.json(df_dict, path)

    @staticmethod
    def dict_as_json(d, path):
        WriteHelper.json(d, path)

    @staticmethod
    def list_as_json(l, path):
        WriteHelper.json(l,path)
        
    @staticmethod
    def as_json(x, path):
        WriteHelper.json(x, path)
        

    @staticmethod
    def list_as_lines(lines, path, line_separator = "\n", header:str = None):
        contents = str.join(line_separator, lines)
        if header is not None:
            contents = header + "\n" + contents
        WriteHelper.write(path, contents)
        
    @staticmethod
    def write_lines(path:str, lines, line_separator = "\n", header:str = None):
        WriteHelper.list_as_lines(lines, path, line_separator, header)

    @staticmethod
    def write(path:str, data):

        # if "~/" in path:
        #     path = _project.from_root(path.split("~/")[1])

        os.makedirs(os.path.dirname(path), exist_ok=True)

        f = open(path, "w")
        f.write(data)
        f.close()
        
    @staticmethod
    def json_readable(obj, path):
        if not path.endswith(".json"):
            path += ".json"

        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = jsonpickle.encode(obj, unpicklable=False)
        data_obj = json.loads(data)
        formatted_data = json.dumps(data_obj, indent=4)

        with open(path, "w") as f:
            f.write(formatted_data)