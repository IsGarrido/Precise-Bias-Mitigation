import re

class RegexHelper:

    @staticmethod
    def label_to_foldername(text:str) -> str:
        transformed_text = re.sub(r'[^\w\s-]', '', text)  # Remove non-alphanumeric characters
        transformed_text = transformed_text.replace(' ', '_')  # Replace spaces with underscores
        return transformed_text
