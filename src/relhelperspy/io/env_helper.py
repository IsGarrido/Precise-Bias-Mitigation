import os

class EnvironmentHelper:

    __is_test_env = None

    @staticmethod
    def is_test_env() -> bool:
        
        # cache
        if EnvironmentHelper.__is_test_env is None:
            EnvironmentHelper.__is_test_env = os.path.expanduser('~') == "/home/isgarrido" # TODO: improve this
        
        return EnvironmentHelper.__is_test_env