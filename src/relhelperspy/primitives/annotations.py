import datetime as dt
import inspect
import os
from relhelperspy.io.env_helper import EnvironmentHelper as _env

def fail_safe_dev(fn):

    if type(fn).__name__ == 'staticmethod':
        fn = fn.__func__
    
    # https://stackoverflow.com/questions/50673566/how-to-get-the-path-of-a-function-in-python
    file = 'file.py'
    try:
        file =os.path.abspath(inspect.getfile(fn)).split('/')[-1]
    except:
        pass

    def wrapper(df, *args, **kwargs):

        if _env.is_test_env():
            return fn(df, *args, **kwargs)
        else:
            try:
                return fn(df, *args, **kwargs)
            except Exception as e:
                print(f"{file} | {fn.__name__} failed, returning None. \nError:\n{e}\n")
                return None

    return wrapper

def fail_safe(fn):
    if type(fn).__name__ == 'staticmethod':
        fn = fn.__func__
    
    # https://stackoverflow.com/questions/50673566/how-to-get-the-path-of-a-function-in-python
    file = 'file.py'
    try:
        file =os.path.abspath(inspect.getfile(fn)).split('/')[-1]
    except:
        pass

    def wrapper(df, *args, **kwargs):
        try:
            return fn(df, *args, **kwargs)
        except Exception as e:
            print(f"{file} | {fn.__name__} failed, returning None. \nError:\n{e}\n")
            return None

    return wrapper


def log_time(fn):

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


def log_sandwich(fn):

    if type(fn).__name__ == 'staticmethod':
        fn = fn.__func__
    
    # https://stackoverflow.com/questions/50673566/how-to-get-the-path-of-a-function-in-python
    file = 'file.py'
    try:
        file =os.path.abspath(inspect.getfile(fn)).split('/')[-1]
    except:
        pass

    def wrapper(df, *args, **kwargs):
        print(f"[Sandwich v] {file} | {fn.__name__}")
        result = fn(df, *args, **kwargs)
        print(f"[Sandwich ^] {file} | {fn.__name__}")
        return result
    
    return wrapper


# https://stackoverflow.com/questions/30382556/python-count-number-of-times-function-passes-through-decorator
log_time_with_counter_calls = {}

def log_time_with_counter(fn):

    if type(fn).__name__ == 'staticmethod':
        fn = fn.__func__
    
    # https://stackoverflow.com/questions/50673566/how-to-get-the-path-of-a-function-in-python
    file = 'file.py'
    try:
        file =os.path.abspath(inspect.getfile(fn)).split('/')[-1]
    except:
        pass

    def wrapper(df, *args, **kwargs):

        uid = file + "_" + fn.__name__
        if uid in log_time_with_counter_calls:
            log_time_with_counter_calls[uid] += 1
        else:
            log_time_with_counter_calls[uid] = 1

        tic = dt.datetime.now()
        result = fn(df, *args, **kwargs)
        toc = dt.datetime.now()
        print(f"[{log_time_with_counter_calls[uid]}] {file} | {fn.__name__} took {toc - tic }")
        return result
    return wrapper

log_time_with_counter_every_1000_calls_counter = {}
log_time_with_counter_every_1000_calls_time = dt.datetime.now()

def log_time_with_counter_every_1000_calls(fn):

    if type(fn).__name__ == 'staticmethod':
        fn = fn.__func__
    
    # https://stackoverflow.com/questions/50673566/how-to-get-the-path-of-a-function-in-python
    file = 'file.py'
    try:
        file =os.path.abspath(inspect.getfile(fn)).split('/')[-1]
    except:
        pass

    def wrapper(df, *args, **kwargs):

        uid = file + "_" + fn.__name__
        if uid in log_time_with_counter_every_1000_calls_counter:
            log_time_with_counter_every_1000_calls_counter[uid] += 1
        else:
            log_time_with_counter_every_1000_calls_counter[uid] = 1

        result = fn(df, *args, **kwargs)

        if log_time_with_counter_every_1000_calls_counter[uid] % 1000 == 0:

            tic = log_time_with_counter_every_1000_calls_time
            toc = dt.datetime.now()
            print(f"[{log_time_with_counter_every_1000_calls_counter[uid]}] {file} | {fn.__name__} took {toc - tic }")
            tic = toc

        return result
    return wrapper


log_time_with_counter_every_N_calls_counter = {}
log_time_with_counter_every_N_calls_time = dt.datetime.now()

def log_time_with_counter_every_N_calls(num_calls):
    def decorator(fn):
        if type(fn).__name__ == 'staticmethod':
            fn = fn.__func__

        file = 'file.py'
        try:
            file = os.path.abspath(inspect.getfile(fn)).split('/')[-1]
        except:
            pass

        def wrapper(df, *args, **kwargs):
            uid = file + "_" + fn.__name__
            if uid in log_time_with_counter_every_N_calls_counter:
                log_time_with_counter_every_N_calls_counter[uid] += 1
            else:
                log_time_with_counter_every_N_calls_counter[uid] = 1

            result = fn(df, *args, **kwargs)

            if log_time_with_counter_every_N_calls_counter[uid] % num_calls == 0:
                tic = log_time_with_counter_every_N_calls_time
                toc = dt.datetime.now()
                elapsed_time = toc - tic
                elapsed_seconds = elapsed_time.total_seconds()
                elapsed_seconds_rounded = round(elapsed_seconds, 2)
                print(f"[{log_time_with_counter_every_N_calls_counter[uid]}] {file} | {fn.__name__} took {elapsed_time} (+{elapsed_seconds_rounded} seconds)")
                tic = toc

            return result

        return wrapper

    return decorator