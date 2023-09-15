import json
from typing import Dict, Any

class TaskBase:
    def __init__(self):
        self._task_data = {}
    
    def get_subtask_data(self, name):
        if name in self._task_data:
            return self._task_data[name]
        subtask = self.build_subtask(name)
        self._task_data[name] = subtask
        return subtask
    
    def evaluate_outputs(self, outputs, name) -> Dict[str, Any]:
        raise NotImplementedError
    
    def get_all_subtask(self):
        return []
    
    def cut_text(self, text, max_len):
        token_ids = self.tokenizer(text, truncation = True, max_length = max_len)
        out_text = self.tokenizer.decode(
            token_ids.input_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces = True, # if false, chinese characters are splited by space
        )
        return out_text