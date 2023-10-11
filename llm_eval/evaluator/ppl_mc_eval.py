"""
Perplexity Evaluator for Multi-choice questions.
"""
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import List, Optional, Union
import time
import json


from llm_eval.handler.multi_choice_ppl import MultiChoicePPL
from llm_eval.task import AutoTask
from llm_eval.utils import read_jsonl, read_json
from llm_eval.task.ljp import JudgmentPredictionConfig, JudgmentPrediction_Task
from .utils import perform_task_timely_save

class MultiChoice_Pipeline:
    """
    Manage multi subtask evaluate, save results and metrics

    Args:
        run_config:
            agent_config: 
                model
                trust_remote_code
            task_config:
                
    """
    def __init__(self, output_dir, run_config, late_init = True):
        self.output_dir = Path(output_dir)
        self.run_config = deepcopy(run_config)
        self.worker = MultiChoicePPL(run_config['model_config'], late_init = late_init)
        self.task = AutoTask.from_dict(run_config['task_config'], tokenizer = self.worker.tokenizer)
    
    def set_output_dir(self, path):
        self.output_dir = Path(path)
    
    def set_config(self, config):
        self.run_config = deepcopy(config)

    def map_func(self, example):
        output = self.worker(example)
        output = {k:example[k] for k in ['idx', 'question', 'options', 'label']} | output
        return output

    def do_eval(self, sub_tasks: Union[str, List[str]]):
        if isinstance(sub_tasks, str):
            if sub_tasks == 'all':
                sub_tasks = self.task.get_all_subtask()
            else:
                sub_tasks = sub_tasks.split(',')
        self.output_dir.mkdir(parents = True, exist_ok = True)
        metric_path = self.output_dir / "eval_results.txt"
        for sub_t in sub_tasks:
            print(f'Evaluate subtask: {sub_t}')
            sub_t_dir = self.output_dir / sub_t
            sub_t_dir.mkdir(parents = True, exist_ok = True)
            raw_output_path = sub_t_dir / 'raw_output.txt'
            with open(sub_t_dir / 'run_config.json', 'w') as f:
                json.dump(self.run_config, f, indent = 4, ensure_ascii=False)

            task_data = self.task.get_subtask_data(sub_t)

            perform_task_timely_save(task_data, self.map_func, raw_output_path)
            
            llm_outputs = read_jsonl(raw_output_path)

            try:
                metrics = self.task.evaluate_outputs(llm_outputs, sub_t)
            except Exception as e:
                print(f'Error during evaluation. {str(e)}')
                continue
            metrics = {k:float(v) for k,v in metrics.items()} # for serialization
            record = {'time': time.time(), 'subtask': sub_t, 'metrics': metrics}
            log = json.dumps(record, ensure_ascii='False')
            print(log)
            with open(metric_path, 'a', encoding='utf8') as f:
                f.write(log + '\n')
    
    def load_peft_model(self, model_id):
        self.worker.load_peft_model(model_id)
        # update model config
        self.run_config['model_config']['model_name'] = model_id
    
    def change_output_dir(self, new_path):
        self.output_dir = new_path