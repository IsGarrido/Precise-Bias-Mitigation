import os
from pathlib import Path
from relhelperspy.io.read_helper import ReadHelper as _read
from relhelperspy.io.write_helper import WriteHelper as _write

class ProjectHelper:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_project_root() -> str:
        path = Path(__file__).as_posix()
        src_index = path.index('/src/')
        return path[0:src_index]

    @staticmethod
    def from_root(*paths):
        root = ProjectHelper.get_project_root()
        return os.path.join(root, *paths)

    @staticmethod
    def special_path(main_type: str, sub_type: str, experiment: str, folder: str = None, file: str = None):
        
        path = ProjectHelper.from_root(main_type)

        if sub_type is not None:
            path = os.path.join(path, sub_type)

        if experiment is not None:
            path = os.path.join(path, experiment)

        if folder is not None:
            path = os.path.join(path, folder)

        if file is not None:
            return os.path.join(path, file)

        return path

    @staticmethod
    def data_path(folder = None, file = None) -> str:
        return ProjectHelper.special_path('assets', 'data', None, folder, file)

    @staticmethod
    def test_data_path(folder = None, file = None) -> str:
        return ProjectHelper.special_path('assets', 'tests', None, folder, file)

    @staticmethod
    def result_path(experiment = None, folder = None, file = None) -> str:
        return ProjectHelper.special_path('assets', 'result', experiment, folder, file)
        
    @staticmethod
    def get_last_run_index(experiment = None) -> int:
        run_file_path = ProjectHelper.result_path(experiment, file="run_index.txt")
        try:
            current_run_index = int(_read.as_text(run_file_path))
            return current_run_index
        except:
            pass
        return -1
    
    @staticmethod
    def new_run(experiment = None) -> None:
        last_run_index = ProjectHelper.get_last_run_index(experiment)
        current_run_index = last_run_index + 1
        _write.write_text(str(current_run_index), ProjectHelper.result_path(experiment, file="run_index.txt"))
        return current_run_index