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

from llm_eval.agent import AutoAgent, AgentArguments, GenerateArguments, GenerateAgentBase
from llm_eval.utils import read_jsonl
from llm_eval.task import AutoTask
from llm_eval.task.ljp import JudgmentPredictionConfig, JudgmentPrediction_Task

class Evaluator:
    """
    Manage multi subtask evaluate, save results and metrics
    """
    def __init__(self, output_dir, agent_args, gen_args, task_args):
        self.output_dir = Path(output_dir)
        self.agent = AutoAgent.from_config(agent_args, gen_args)
        self.task = AutoTask.from_config(task_args.__dict__, tokenizer = self.agent.tokenizer)
    
    def do_eval(self, sub_tasks: Union[str, List[str]]):
        if isinstance(sub_tasks, str):
            if sub_tasks == 'all':
                sub_tasks = self.task.get_all_subtask()
            else:
                sub_tasks = [sub_tasks]
        self.output_dir.mkdir(parents = True, exist_ok = True)
        metric_path = self.output_dir / "eval_results.txt"
        for sub_t in sub_tasks:
            sub_t_dir = self.output_dir / sub_t
            sub_t_dir.mkdir(parents = True, exist_ok = True)
            raw_output_path = sub_t_dir / 'raw_output.txt'

            task_data = self.task.get_subtask_data(sub_t)
            self.agent.infer_all(task_data, save_path = raw_output_path)
            llm_outputs = read_jsonl(raw_output_path)

            metrics = self.task.evaluate_outputs(llm_outputs, sub_t)
            metrics = {k:float(v) for k,v in metrics.items()} # for serialization
            record = {'time': time.time(), 'subtask': sub_t, 'metrics': metrics}
            log = json.dumps(record, ensure_ascii='False')
            print(log)
            with open(metric_path, 'a') as f:
                f.write(log + '\n')

def main(args):
    pass

if __name__ == '__main__':
    from argparse import ArgumentParser
    from transformers import HfArgumentParser
    parser = HfArgumentParser([AgentArguments, GenerateArguments, JudgmentPredictionConfig])
    parser.add_argument('output_dir')
    parser.add_argument('--sub_tasks', type = str)
    agent_args, gen_args, task_args, args = parser.parse_args_into_dataclasses()
    
    worker = Evaluator(args.output_dir, agent_args, gen_args, task_args)
    worker.do_eval(args.sub_tasks)
