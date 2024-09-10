from typing import List
from relhelperspy.io.project_helper import ProjectHelper
from relhelperspy.io.read_helper import ReadHelper as _read
from relhelperspy.io.rel_project_helper import RelProjectHelper
from relhelperspy.primitives.string_helper import StringHelper as _string

class TrainLayerCommon:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def get_source_sentences(experiment:str, train_data_folder: str) -> List[str]:
        _project = RelProjectHelper(experiment)
        path = _project.get_path(train_data_folder, "sentences.txt")
        source_sentences = _read.read_lines(path)
        return source_sentences
    
    @staticmethod
    def get_target_words(experiment:str, train_data_folder: str) -> List[str]:
        _project = RelProjectHelper(experiment)
        path = _project.get_path(train_data_folder, "words.txt")
        target_words = _read.read_lines(path)
        return target_words
    
    @staticmethod 
    def get_model_save_path(experiment: str, train_folder:str, from_layer: int, to_layer: int = -1) -> str:
        train_folder_as_name = _string.safe_file_name(train_folder)
        if to_layer == -1:
            return ProjectHelper.from_root(".result", experiment, f"{train_folder_as_name}_finetuned_layer_{to_layer}")
        return ProjectHelper.from_root(".result", experiment, f"{train_folder_as_name}_finetuned_layer_{from_layer}_to_{to_layer}")    
    
    @staticmethod
    def project() -> RelProjectHelper:
        return RelProjectHelper("common")
    
    @staticmethod
    def get_slurm_contents(experiment: str, slurm_resource: str, job_name: str = None, command: str = None, memory: int = 48) -> str:
        
        example_eval_path = TrainLayerCommon.project().get_path("generic.example.sbs")
        contents = _read.as_text(example_eval_path)
        
        if job_name is not None:
            contents = contents.replace("$JOB_NAME", job_as_file)
            job_as_file = _string.safe_file_name(job_name)
            contents = contents.replace("$NAME", job_as_file)
            
        if command is not None:
            contents = contents.replace("$COMMAND", command)
            
        contents = contents.replace("$EXPERIMENT", experiment)
        contents = contents.replace("$SLURM_RESOURCE", slurm_resource)
        contents = contents.replace("$MEMORY", str(memory))
        contents = contents.replace("$DEBUG_INFO", f"Using generic.example.sbs as template. Debug info, experiment: {experiment}, job_name: {job_name}, command: {command}, memory: {memory}")
        
        return contents
        
    @staticmethod
    def get_eval_command(experiment: str, train_folder: str, layer: int, model_path: str, to_layer: int = -1) -> str:
        
        cmd = f"src/evaluate_models.py --experiment {experiment} --train_folder {train_folder} --layer {layer} --model_path {model_path}"
        if to_layer != -1:
            cmd += f" --to_layer {to_layer}"
        return cmd

    @staticmethod
    def get_eval_slurm_contents(experiment: str, slurm_resource: str, model_name:str, base_contents: str, from_layer: str, to_layer:int = -1, memory: int = 48):
        
            contents = base_contents
            contents = contents.replace('$EXPERIMENT', experiment) 
            contents = contents.replace('$CURRENT_LAYER', from_layer)
            
            if to_layer != -1:
                contents = contents.replace('$LAYER_WINDOW_LIMIT', str(to_layer) )
                
            contents = contents.replace('$MODEL_NAME', model_name)
            contents = contents.replace('$SLURM_RESOURCE', slurm_resource)
            
            contents = contents.replace("$MEMORY", str(memory))
            contents = contents.replace("$DEBUG_INFO", f"Using generic.example.sbs as template. Debug info, experiment: {experiment}, model_name: {model_name}, from_layer: {from_layer}, to_layer: {to_layer}, memory: {memory}")
            
            return contents