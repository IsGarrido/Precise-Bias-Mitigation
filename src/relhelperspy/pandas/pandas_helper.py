import types
import os
import inspect
import pandas as pd
import datetime as dt
import sqlite3
import sqlalchemy
from relhelperspy.io.write_helper import WriteHelper as _write
from relhelperspy.io.memory_helper import MemoryHelper as _memory
from relhelperspy.text.ColorHelper import ColorHelper as _color
from relhelperspy.io.read_helper import ReadHelper as _read

def log_pd(fn):

    if type(fn).__name__ == 'staticmethod':
        fn = fn.__func__
    
    # https://stackoverflow.com/questions/50673566/how-to-get-the-path-of-a-function-in-python
    file = 'file.py'
    try:
        file =os.path.abspath(inspect.getfile(fn)).split('/')[-1]
    except:
        pass

    def wrapper(df, *args, **kwargs):
        tic = dt.datetime.now()
        result = fn(df, *args, **kwargs)
        toc = dt.datetime.now()
        print(f"{file} | {fn.__name__} took {toc - tic }")
        return result
    return wrapper
    
class PandasHelper:

    @staticmethod
    def from_classes(items) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in items])

    @log_pd
    @staticmethod
    def read_tsv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep='\t', header = 0)
        try:
            df = PandasHelper.remove_comments(df)
        except:
            _color.print_italic("Error removing comments")
        return df
    
    @log_pd
    @staticmethod
    def read_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, header = 0)
        return PandasHelper.remove_comments(df)

    @staticmethod
    def remove_comments(df) -> pd.DataFrame:
        first_col = df.columns[1]
        return df[~df[first_col].str.contains("#")]

    # https://stackoverflow.com/questions/39475978/apply-function-to-each-cell-in-dataframe
    @log_pd
    @staticmethod
    def apply_all_cells(df, fn) -> pd.DataFrame:
        return df.applymap(fn)
    
    @log_pd
    @staticmethod
    def apply_column(df, col, fn) -> pd.DataFrame:
        return df[col].apply(fn)

    # @log_pd
    @staticmethod
    def from_dict(items: dict):
        return pd.DataFrame.from_dict(items)
    
    @staticmethod
    def from_dict_with_index(items:dict):
        return pd.DataFrame.from_dict([items])
    
    @staticmethod
    def from_list(items: list):
        return pd.DataFrame(items)

    @staticmethod
    def to_dict(df: pd.DataFrame):
        return df.to_dict('records')

    # @log_pd
    @staticmethod
    def remove_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df.drop(col, axis=1)
    
    # @log_pd
    @staticmethod
    def remove_col_idx(df: pd.DataFrame, idx: int) -> pd.DataFrame:
        return df.drop(df.columns[idx], axis=1)
    
    @log_pd
    def save(df: pd.DataFrame, path: str):
        PandasHelper.save_csv(df, path)

    @log_pd
    def save_csv(df: pd.DataFrame, path: str):
                
        if not path.endswith(".csv"):
            path = path + ".csv"

        df.to_csv(path, sep="\t")
        print("Saved " + path )

    @log_pd
    def save_tsv(df: pd.DataFrame, path: str):
        if not path.endswith(".tsv"):
            _color.print_italic("No tsv extension found. Adding it.")
            path = path + ".tsv"

        df.to_csv(path, sep="\t")
        print("Saved " + path )

    @log_pd
    def save_json(df: pd.DataFrame, path: str):
        if not path.endswith(".json"):
            path = path + ".json"

        df.to_json(path)
        print("Saved " + path )

    @log_pd
    @staticmethod
    def save_sqlite(df: pd.DataFrame, path: str):

        if not path.endswith(".sqlite"):
            path = path + ".sqlite"
            
        _write.delete_file(path)
        cnx = sqlite3.connect(path)
        df.to_sql(name='data', con=cnx)

    @log_pd
    @staticmethod
    def load_smart(path: str):
        if path.endswith(".feather"):
            return PandasHelper.load_feather(path)
        if path.endswith(".json"):
            return PandasHelper.load_json(path)
        if path.endswith(".tsv"):
            return PandasHelper.read_tsv(path)
        elif path.endswith(".csv"):
            return PandasHelper.read_csv(path)
        elif path.endswith(".txt"):
            return _read.read_as_list(path)

        return PandasHelper.load_smart(path + ".feather")
        
    @log_pd
    @staticmethod
    def load_sqlite(path: str) -> pd.DataFrame:
        engine = sqlalchemy.create_engine('sqlite:///' + path) # ensure this is the correct path for the sqlite file. 
        connection = engine.raw_connection()
        df = pd.read_sql('SELECT * FROM data' ,connection)
        return df

    @log_pd
    @staticmethod
    def save_feather(df: pd.DataFrame, path: str):
        
        if not path.endswith(".feather"):
            path = path + ".feather"

        df.to_feather(path)
    


    @log_pd
    @staticmethod
    def load_feather(path: str):
        return pd.read_feather(path)
    
    @log_pd
    def load(path: str) -> pd.DataFrame:
        return pd.read_csv(path, sep = "\t")

    @log_pd
    def load_json(path: str) -> pd.DataFrame:
        return pd.read_json(path); 

    @log_pd
    @staticmethod
    def log(pd: pd.DataFrame) -> pd.DataFrame:
        return pd
    
    # Explore
    @staticmethod
    def unique_by_col(df: pd.DataFrame, col: str):
        return df[col].unique()
    
    @staticmethod
    def unique_count_by_col(df: pd.DataFrame, col: str):
        return df[col].value_counts()

    @staticmethod
    def get_io_stats(df: pd.DataFrame):

        memory = df.memory_usage(index=True).sum()
        memory_statement = _memory.convert_bytes_to_human(memory)

        return memory_statement

    @staticmethod
    def deep_copy(df: pd.DataFrame):
        return df.copy(deep=True)

    @log_pd
    @staticmethod
    def expand_array_col_into_rows(df: pd.DataFrame, col: str):
        return df.explode(col)

    @log_pd
    @staticmethod
    def expand_object_col_into_columns(df: pd.DataFrame, col: str):
        # return df[col].apply(pd.Series)
        return df.join(pd.DataFrame(df.pop(col).values.tolist()))
    
    @log_pd
    @staticmethod
    def col_as_list(df: pd.DataFrame, col: str):
        return df[col].values.tolist()
    
    @log_pd
    @staticmethod
    def join_on_cols(df1: pd.DataFrame, df2: pd.DataFrame, cols: list):
        return df1.merge(df2, on=cols)
    
    @log_pd
    @staticmethod
    def join_lr_col(df1: pd.DataFrame, df2: pd.DataFrame, col_left: str, col_right: str):
        return df1.merge(df2, left_on=col_left, right_on=col_right)