"""
Multi choice question perplexity evaluation of LJP.

There are two ways to formulate the question:
    - the question do not include options
    - the question includes all options
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
from typing import Any, Dict, Optional, List
from datasets import load_dataset
from functools import partial
from sklearn.metrics import accuracy_score

from llm_eval.task.base import TaskBase
from llm_eval.utils import read_jsonl, read_json
from llm_eval.parse import parse_bm25_all, get_classification_metrics
from llm_eval.task.ljp_mc import JudgPredMulti_Task

"""
prompt_config_file is a dict of 
    - **free_prompt**: with the slot of `input`
    - **full_prompt**: with the slot of `input` and `options`
"""
@dataclass
class JudgPredMulti_Config:
    data_path: Optional[str] = None
    prompt_config_file: Optional[str] = None
    task_type: str = 'ljp_mc_ppl'
    query_max_len: int = 1000
    demo_max_len: int = 400


class JudgPredMulti_PPL_Task(JudgPredMulti_Task):
    def clean_example_label(self, example):
        example['charge'] = self._clean_label(example['charge'])
        example['cdd_charge_list'] = list(map(self._clean_label, example['cdd_charge_list']))
        example['label'] = example['cdd_charge_list'].index(example['charge'])
        return example
    def load_data(self, data_path = None):
        data_path = data_path or self.config.data_path
        data_path = Path(data_path)
        print(f'Load data from {data_path}')
        test_ds = load_dataset('json', data_files = str(data_path / 'test_data.json'), split = 'train')
        test_ds = test_ds.map(
            self.clean_example_label, 
            load_from_cache_file = False, 
            num_proc = 5
        )
        self.test_ds = test_ds
    
    def get_all_subtask(self):
        return '5opt-add', '5opt-free'
    
    def build_options(self, labels):
        n = len(labels)
        ord_c = [chr(ord('A') + i) for i in range(n)]
        opt_strs = [f'({o}) {l}' for o, l in zip(ord_c, labels)]
        return opt_strs

    def build_example(self, example, add_options = False):
        """
        Build the question and options field.
        """
        tpl = self.prompt_config['full_prompt'] if add_options else self.prompt_config['free_prompt']
        kws = {'input': self.cut_text(example['fact'], self.config.query_max_len)}
        if add_options:
            opt_str = ' '.join(self.build_options(example['cdd_charge_list']))
            kws['options'] = opt_str
        quest = tpl.format(**kws)
        return {'question': quest, 'options': example['cdd_charge_list']}

    
    def build_subtask(self, name):
        """
        Processed Data format:
        Dict of :
            - question (`str`)
            - options (`List[str]`)
            - label (`int`): ground truth option id
        """
        if name == '5opt-add':
            add_options = True
        elif name == '5opt-free':
            add_options = False
        else:
            raise ValueError(f'unrecognized subtask name: {name}')
        test_ds = self.test_ds.map(
            partial(self.build_example, add_options = add_options),
            load_from_cache_file = False, 
            num_proc = 5
            )
        test_ds = test_ds.with_format(
            columns=["idx", "question", "options", "label"]
        )
        return test_ds.to_list()
        
    def evaluate_outputs(self, outputs: List[Dict], name=None) -> Dict[str, Any]:
        task_data = self.get_subtask_data(name)
        grounds = [k['label'] for k in task_data]
        idx2outputs = {k['idx']: k for k in outputs}
        ordered_outputs = [idx2outputs[k['idx']] for k in task_data]

        metric_fields = [k for k in ordered_outputs[0] if k.startswith('choice')]
        results = {}
        for met_f in metric_fields:
            met_name = re.match(r'choice_(.*)', met_f).group(1)
            preds = [k[met_f] for k in ordered_outputs]
            acc = accuracy_score(grounds, preds)
            results['acc_' + met_name] = acc
        return results
    