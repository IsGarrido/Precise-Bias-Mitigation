import re
import time
import unidecode

class StringHelper:

    def __init__(self) -> None:
        pass

    def as_file_name(text : str):
        t = text.replace(" ", "_")

        # DEJAR [MASK] como MASK
        t = t.replace("[", "")
        t = t.replace("]", "")

        # Quitar slash del nombre del modelo
        t = t.replace("/", "_")

        # Quitar puntos
        t = t.replace(".", "")
        t = t.replace(",", "")
        t = t.replace(":", "")

        return t
    
    def safe_file_name(text: str) -> str:
        # Replace spaces with underscores and remove special characters
        # Special characters to remove: [ ] / . , :
        text = re.sub(r'[ \[\]/.,:]', lambda match: '_' if match.group(0) == ' ' else '', text)
        return text
    
    def as_log_file_name(text : str, extension: str = None, include_timestamp: bool = True):
        
        filename = StringHelper.as_file_name(text)
        
        if include_timestamp:
            filename = f"{filename}_{int(time.time())}"
        
        if extension is not None:
            filename = f"{filename}.{extension}"
        
        return filename

    def from_int(val: int, leading_zeroes = 0):
        s: str = str(val)
        if leading_zeroes > 0:
            return s.zfill(leading_zeroes)
        else:
            return s
        
    def cut_on_sentence_end(text: str):
        punctuation = {'.', '?', '\n'}
        next_punctuation_index = -1
        for index, char in enumerate(text):
            if index >= 20 and char in punctuation:
                next_punctuation_index = index
                break
        if next_punctuation_index != -1:
            return text[:next_punctuation_index]
        else:
            return text

    def copy(text: str):
        return (text + '.')[:-1]
    
    def replace_accents_with_non_accents(text: str):
        return unidecode.unidecode(text)

    @staticmethod
    def join(items: list, separator: str) -> str:
        return separator.join(map(str, items))