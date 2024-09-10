import json
import pandas as pd
from typing import TextIO
from os.path import exists
import hashlib
import os

class ReadHelper:
    
    def __init__(self) -> None:
        pass

    def as_text(path) -> str:
        file = open(path, "r")
        content = file.read()
        return content
    
    @staticmethod
    def json_as_dict(path) -> 'dict[str, str]':
        # Define the maximum path length for a filename
        max_filename_length = 255  # Maximum filename length for most filesystems

        # Check if the filename is too long
        if len(os.path.basename(path)) > max_filename_length:
            # If the filename is too long, handle it as a hashed filename
            name, ext = os.path.splitext(os.path.basename(path))
            shortened_name = hashlib.md5(name.encode()).hexdigest() + ext
            path = os.path.join(os.path.dirname(path), shortened_name)

        # Read and parse the JSON file
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        
        return data
    
    def json_as_list(path) -> 'list[str]':
        with open(path) as json_file:
            data = json.load(json_file)
            return data

    def pandad_read_tsv(file):
        data = pd.read_csv(file, sep='\t', decimal=",")
        return data
    
    def read_smart(path: str):
        # if the file has a .json extension, we read it as json
        if path.endswith(".tsv"):
            res = ReadHelper.read_tsv(path)
        elif path.endswith(".csv"):
            res = ReadHelper.read_csv(path)
        else:
            raise Exception("Unknown file extension " + path)
        
        # if all elements of the inner lists have only one elements, lets flatten the list
        if type(res) == list:
            if all(len(inner_lst) == 1 for inner_lst in res):
                res = [element for inner_lst in res for element in inner_lst]

        return res

    def read_lines(path:str) -> 'list[str]':
        return ReadHelper.read_lines_as_list(path)
        
    def read_lines_as_list(path: str) -> 'list[str]':
        file = open(path, "r")
        content = file.read()
        lines: list[str] = content.split("\n")
        return lines

    def read_lines_as_dict(path):
        items = ReadHelper.read_lines_as_list(path)
        hashmap = dict.fromkeys(items, True)
        return hashmap
    
    def json_list_as_lookup(path):
        items = ReadHelper.json_as_list(path)
        hashmap = dict.fromkeys(items, True)
        return hashmap

    def read_paired_tsv(file):
        data = ReadHelper.read_lines_as_list(file)
        data = filter(lambda line: not line.startswith("#"), data)
        data = filter(lambda line: line.strip() != "", data)

        paired = [item.split("\t") for item in data]
        return paired

    def read_lines_as_col_excel(path: str) -> 'dict[str, list[str]]':
        data: dict[str, list[str]] = {}

        lines: list[str] = ReadHelper.read_lines_as_list(path)
        tag_line: str = 2 if lines[0].startswith("#") else 1
        tags: list[str] = lines[tag_line].split("\t")

        # Preparar las listas vacias
        for tag in tags:
            data[tag] = []

        # Leemos las lines con datos y las vamos pasando a la lista que corresponda
        data_lines = lines[tag_line + 1:]
        for line in data_lines:
            parts = line.split("\t")
            for idx, part in enumerate(parts):
                if part != "":
                    tag = tags[idx]
                    data[tag].append(part)

        return data


    # Deja el objeto de listas de string como un hashmap/lookup
    def read_lines_as_col_excel_asdict(path: str) -> 'dict[str, str]':
        data: dict[str, list[str]] = ReadHelper.read_lines_as_col_excel(path)
        plain_data: dict[str, str] = {}

        for key, list_val in data.items():
            for item in list_val:
                plain_data[item] = key

        return plain_data


    @staticmethod
    def read_raw(path) -> str:
        file: TextIO = open(path, "r")
        content: str = file.read()
        return content
    
    @staticmethod
    def read_key_value(path, separator = '=') -> dict[str, str]:
        data = ReadHelper.read_as_list(path)
        hashmap = dict()
        for item in data:
            parts = item.split(separator)
            hashmap[parts[0].strip()] = parts[1].strip()
        return hashmap

    @staticmethod
    def read_as_list(path: str) -> 'list[str]':
        content = ReadHelper.read_raw(path)
        lines = content.split("\n")
        filtered: filter[str] = filter(lambda line: not line.startswith("#"), lines)
        filtered = filter(lambda line: line.strip() != "", filtered)

        return list(filtered)

    @staticmethod
    def read_as_dict(path: str) -> 'dict[str]':
        items = ReadHelper.read_as_list(path)
        hashmap = dict.fromkeys(items, True)
        return hashmap

    @staticmethod
    def read_tsv(file: str, remove_header = False) -> 'list[list[str]]':
        data: list[str] = ReadHelper.read_as_list(file)
        filtered: filter[str] = filter(lambda line: not line.startswith("#"), data)
        filtered = filter(lambda line: line.strip() != "", filtered)
        
        if remove_header:
            next(filtered)

        paired: list[list[str]] = [item.split("\t") for item in filtered]
        return paired    
    
    @staticmethod
    def get_first_existing_file(files: 'list[str]') -> str:
        for file in files:
            if exists(file):
                return file
        raise Exception("No file exists in " + str(files))
    
    @staticmethod
    def pandas_read_csv(file):
        data = pd.read_csv(file, decimal=",")
        return data