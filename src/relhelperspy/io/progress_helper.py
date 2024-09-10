
from relhelperspy.io.read_helper import ReadHelper as _read
from relhelperspy.io.write_helper import WriteHelper as _write
from relhelperspy.io.project_helper import ProjectHelper as _project
from relhelperspy.text.ColorHelper import ColorHelper as _color

class ProgressHelper:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def progress_filepath(concept: str):
        name = f"{concept}.progress.txt"
        path = _project.result_path("progress", concept, name)
        return path
        
    @staticmethod
    def write_loop_progress_str(concept: str, items: 'list[str]', current_item: str):
        
        current_item_index = items.index(current_item)
        completed_items = items[:current_item_index+1]
        remaining_items = items[current_item_index+1:]
        
        completed_items_text = ["\t" + item for item in completed_items]
        remaining_items_text = ["\t" + item for item in remaining_items]
        
        concant_completed = "\n" + "\n".join(completed_items_text)
        concant_remaining = "\n" + "\n".join(remaining_items_text)
        
        completed_text = f"Completed: {concant_completed}"
        remaining_text = f"Remaining: {concant_remaining}"
        
        progress_text = completed_text + "\n\n" + remaining_text
        path = ProgressHelper.progress_filepath(concept)
        _write.txt(progress_text, path)
        
        _color.print_path("Progress saved in ", path)
        
    def read_loop_progress_str(concept: str, items: 'list[str]'):
        
        path = ProgressHelper.progress_filepath(concept)
        
        try:
            lines = _read.read_lines_as_list(path)
            lines = [line.strip("\t").strip() for line in lines]
        except FileNotFoundError:
            _color.print_path("Progress file not found, starting from scratch", path)
            current_item = items[0]
            return current_item, 0
        
        remaining_index = lines.index("Remaining:")
        current_item = lines[remaining_index+1]
        current_item_index = items.index(current_item)
                
        _color.print_path(f"Progress recovered, {current_item_index}/{len(items)} remining", path)

        return current_item, current_item_index
        

        