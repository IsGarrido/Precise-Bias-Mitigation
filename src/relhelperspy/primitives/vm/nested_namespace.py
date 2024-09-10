from types import SimpleNamespace

class NestedNamespace(SimpleNamespace):
    
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            elif isinstance(value, list):
                self.__setattr__(key, map(NestedNamespace, value))
            else:
                self.__setattr__(key, value)
