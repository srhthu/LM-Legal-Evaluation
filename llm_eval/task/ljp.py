"""
Generate prompts for legal judgment prediction
"""

import re
from pathlib import Path
import pandas as pd
import json
import random
import numpy as np
import pickle
from collections import Counter
from transformers import AutoTokenizer
from dataclasses import dataclass
import jieba

from llm_eval.task.base import TaskBase
from llm_eval.utils import read_jsonl, read_json
from llm_eval.parse import parse_bm25_all, get_classification_metrics

@dataclass
class JudgmentPredictionConfig:
    prompt_config_file: str
    meta_data_path: str
    train_data_path: str
    test_data_path: str
    label_path: str
    task_type: str = 'ljp'
    query_max_len: int = 1000
    demo_max_len: int = 400

class JudgmentPrediction_Task(TaskBase):
    """
    {free, multi}-{0..5}-shot
        - free is free generation, multi is multi-choice question
    """
    def __init__(self, config:JudgmentPredictionConfig, tokenizer):
        self.config = config

        self.prompt_config = read_json(config.prompt_config_file)
        self.meta_data = read_jsonl(config.meta_data_path)
        self.train_ds = read_jsonl(config.train_data_path)
        self.test_ds = {k['idx']:k for k in read_jsonl(config.test_data_path)}
        self.tokenizer = tokenizer
        self.load_label()

        self._task_data = {}
    
    def _convert_label(self, lab2id):
        lab2id = {re.sub(r'[\[\]]', '',k): v for k,v in lab2id.items()}
        return lab2id
    
    def load_label(self):
        lab2id = read_json(self.config.label_path)
        self.label2id = self._convert_label(lab2id)
        self.id2label = {v:k for k,v in self.label2id.items()}
    
    def get_all_subtask(self):
        return [f'{t}-{i}-shot' for t in ['free', 'multi'] for i in range(5)]
    
    def build_demo(self, input, input_len, output = None):
        p_config = self.prompt_config
        input = self.cut_text(input, input_len)

        demo_str = '{}: {}\n{}: '.format(
            p_config['query_prompt'], input, p_config['answer_prompt']
        )
        if output is not None:
            demo_str += output
        return demo_str
    
    def build_subtask(self, name):
        ttype = name.split('-')[0]
        n_shot = int(name.split('-')[1])
        p_config = self.prompt_config

        shot_s = 'zs' if n_shot == 0 else 'fs'
        instruct = p_config[f'instruction_{ttype}_{shot_s}']

        task_data = []
        for example in self.meta_data:
            prompt = ''
            prompt += instruct
            
            # add label candidate list
            if ttype == 'multi':
                cdd_label_text = [self.id2label[k] for k in example['cdd_label_ids']]
                prompt += '\n' + '\n'.join(cdd_label_text)
            prompt += '\n\n'

            # add demonstrations
            demo_ids = example[f'{ttype}_demo_ids'][:n_shot]
            demo_examples = [self.train_ds[k] for k in demo_ids]
            for demo_exa in demo_examples:
                demo_str = self.build_demo(
                    demo_exa['text'], 
                    input_len = self.config.demo_max_len, 
                    output = self.id2label[demo_exa['label_charge']]
                )
                prompt += demo_str + '\n\n'

            # add query example
            query_str = self.build_demo(
                self.test_ds[example['idx']]['text'],
                input_len = self.config.query_max_len
            )
            prompt += query_str
            task_data.append({'idx': example['idx'], 'prompt': prompt})
        
        return task_data
    
    def evaluate_outputs(self, outputs, name = None):
        """
        Each output is a dict with fields: `idx`, `choices`
        """
        idx2outputs = {k['idx']: k['choices'] for k in outputs}
        ord_outputs = [idx2outputs[k['idx']] for k in self.meta_data]

        # parse output text to label id
        def _cut(text):
            return list(jieba.cut(text, cut_all = True))
        preds = parse_bm25_all(ord_outputs, self.id2label, _cut)
        grounds = list(map(lambda k: k['label'], self.meta_data))

        return get_classification_metrics(preds, grounds)


if __name__ == '__main__':
    from transformers import HfArgumentParser, AutoTokenizer
    parser = HfArgumentParser([JudgmentPredictionConfig])
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm2-6b', trust_remote_code = True)
    worker = JudgmentPrediction_Task(args, tokenizer)
    data = worker.get_subtask_data('multi-1-shot')
    print(len(data))
    print(data[0])