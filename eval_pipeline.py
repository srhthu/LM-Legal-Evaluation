"""
The pipeline contains:
    - initialize agent
    - build Task object
    - infer on multiple subtasks, save outputs
    - calculate metrics and save to files
"""
from typing import List, Optional, Union
from pathlib import Path
import time
import json
import os
import torch
from copy import deepcopy

from llm_eval.agent import AutoAgent, AgentArguments, GenerateArguments, GenerateAgentBase
from llm_eval.utils import read_jsonl, read_json
from llm_eval.task import AutoTask
from llm_eval.task.ljp import JudgmentPredictionConfig, JudgmentPrediction_Task

class Evaluator:
    """
    Manage multi subtask evaluate, save results and metrics
    """
    def __init__(self, output_dir, run_config):
        self.output_dir = Path(output_dir)
        self.run_config = deepcopy(run_config)
        self.agent = AutoAgent.from_config(
            AgentArguments(**run_config['agent_config']), 
            GenerateArguments(**run_config['generate_config'])
        )
        self.task = AutoTask.from_dict(run_config['task_config'], tokenizer = self.agent.tokenizer)
    
    def do_eval(self, sub_tasks: Union[str, List[str]]):
        if isinstance(sub_tasks, str):
            if sub_tasks == 'all':
                sub_tasks = self.task.get_all_subtask()
            else:
                sub_tasks = sub_tasks.split(',')
        self.output_dir.mkdir(parents = True, exist_ok = True)
        metric_path = self.output_dir / "eval_results.txt"
        for sub_t in sub_tasks:
            sub_t_dir = self.output_dir / sub_t
            sub_t_dir.mkdir(parents = True, exist_ok = True)
            raw_output_path = sub_t_dir / 'raw_output.txt'
            with open(sub_t_dir / 'run_config.json', 'w') as f:
                json.dump(self.run_config, f, indent = 4, ensure_ascii=False)

            task_data = self.task.get_subtask_data(sub_t)
            # for i in range(2):
            #     print(task_data[i]['prompt'])
            # exit()
            self.agent.infer_all(task_data, save_path = raw_output_path)
            llm_outputs = read_jsonl(raw_output_path)

            metrics = self.task.evaluate_outputs(llm_outputs, sub_t)
            metrics = {k:float(v) for k,v in metrics.items()} # for serialization
            record = {'time': time.time(), 'subtask': sub_t, 'metrics': metrics}
            log = json.dumps(record, ensure_ascii='False')
            print(log)
            with open(metric_path, 'a') as f:
                f.write(log + '\n')

def main(output_dir, sub_task, run_config):
    worker = Evaluator(output_dir, run_config)
    worker.do_eval(sub_task)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('--sub_tasks', type = str, help = 'all or subtask names split by ,')
    parser.add_argument('--config', help  = 'config file path.')
    # args to be passed in cmd line
    parser.add_argument('--agent_type', help = 'openai or hf')
    parser.add_argument('--model')
    parser.add_argument('--trust_remote_code', action = 'store_true')

    args = parser.parse_args()

    run_config = read_json(args.config)
    # overwrite some args
    for k in ['agent_type', 'model', 'trust_remote_code']:
        v = vars(args)[k]
        if v is not None and run_config['agent_config'][k] != v:
            run_config['agent_config'][k] = v
            print(f'Overwrite config: {k}={v}')

    main(args.output_dir, args.sub_tasks, run_config)
    
