"""
Multi choice question for LJP. Change the prompt design to indicate the choices.
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

from llm_eval.task.base import TaskBase
from llm_eval.utils import read_jsonl, read_json
from llm_eval.parse import parse_bm25_all, get_classification_metrics

"""
The prompt looks like:
```
You are a legal expert to determin the charges of the defandents. You are given a
case description after <facts>, and you need to output the charge name after <charges>.
Bellow are some examples that show the charges of some cases that are similar to the
case to be determined:
<demos>

Now you are to determin the charge of a specific case. Please choose from the options and your answer should contain the option id and option content:
<options>

<query case>
```
"""

@dataclass
class JudgPredMulti_Config:
    data_path: Optional[str] = None
    prompt_config_file: Optional[str] = None
    task_type: str = 'ljp_mc'
    query_max_len: int = 1000
    demo_max_len: int = 400

class JudgPredMulti_Task(TaskBase):
    """
    The prompt config has the following fields:
    - prompt_template_fs: has slot of *demos* *query*
    - prompt_template_zs: has slot of *query*
    - demo_template: which has the slot of *options*, *input* and *output*
    - query_template: usually same as the demo_template (default to)

    The data folder has:
    - test_data.json
        idx, fact, charge, sim_demo_idx, cdd_charge_list
    - train_data.json
        idx, fact, charge
    - charge2id_clean.json
    """
    def __init__(self, config: JudgPredMulti_Config, tokenizer):
        super().__init__()
        self.config = config

        self.prompt_config = read_json(config.prompt_config_file)
        self.tokenizer = tokenizer
        self.load_data()
    
    def _clean_label(self, label):
        return re.sub(r'[\[\]]', '',label)
    
    def _clean_data_label(self, ds_list):
        for ds in ds_list:
            for d in ds:
                d['charge'] = self._clean_label(d['charge'])

    def load_data(self, data_path = None):
        data_path = data_path or self.config.data_path
        data_path = Path(data_path)
        test_ds = read_jsonl(data_path / 'test_data.json')
        train_ds = read_jsonl(data_path / 'train_data.json')
        self._clean_data_label([test_ds, train_ds])
        for example in test_ds:
            example['cdd_charge_list'] = list(map(self._clean_label, example['cdd_charge_list']))

        idx2train = {k['idx']: k for k in train_ds}

        for example in test_ds:
            example['sim_demo_list'] = [idx2train[idx] for idx in example['sim_demo_idx']]
        self.test_ds = test_ds
        self.label2id = read_json(data_path / 'charge2id_clean.json')
    
    def get_all_subtask(self):
        return [f'mc-{i}-shot' for i in range(5)]
    
    def build_subtask(self, name):
        n_shot = int(name.split('-')[1])
        if n_shot == 0:
            prompts = [self._build_zero_shot_prompt(k) for k in self.test_ds]
        else:
            prompts = [self._build_few_shot_prompt(k) for k in self.test_ds]
        return [
            {
                'idx': example['idx'], 
                'prompt': p
            } for example, p in zip(self.test_ds, prompts)
        ]
    
    def _add_option_order(self, label, ord_id):
        # add option order prefix to label, e.g., (2) label
        prefix_label = f'({ord_id + 1}) ' + label
        return prefix_label

    def build_options(self, labels: List[str])-> List[str]:
        return [self._add_option_order(l, i) for i, l in enumerate(labels)]

    def build_demo(self, input, input_len, output = None, options = None):
        """The demo contains options, input and output"""
        if output is None and 'query_template' in self.prompt_config:
            demo_tpl = self.prompt_config['query_template']
        else:
            demo_tpl = self.prompt_config['demo_template']
        input = self.cut_text(input, input_len)
        output = output if output else ''
        
        return demo_tpl.format(input = input, output = output, options = options)
    
    def build_demo_with_order(self, demo_example, cdd_label_list, demo_max_len):
        """Prepare the options and output option id"""
        try:
            ord_id = cdd_label_list.index(demo_example['charge'])
        except:
            print(demo_example['charge'])
            print(cdd_label_list)
            exit()
        output = self._add_option_order(demo_example['charge'], ord_id)
        labs = '\n'.join(self.build_options(cdd_label_list))
        return self.build_demo(demo_example['fact'], demo_max_len, output, labs)

    def _build_zero_shot_prompt(self, example):
        labs = '\n'.join(self.build_options(example['cdd_charge_list']))
        query = self.build_demo(example['fact'], self.config.query_max_len, None, labs)
        prompt = self.prompt_config['prompt_template_zs'].format(query = query)
        return prompt

    def _build_few_shot_prompt(self, example):
        demos = [
            self.build_demo_with_order(
                d_exp, 
                example['cdd_charge_list'], 
                self.config.demo_max_len
            ) for d_exp in example['sim_demo_list']
        ]
        labs = '\n'.join(self.build_options(example['cdd_charge_list']))
        query = self.build_demo(example['fact'], self.config.query_max_len, None, labs)
        prompt = self.prompt_config['prompt_template_zs'].format(
            demos = '\n\n'.join(demos),
            query = query
        )
        return prompt

    def evaluate_outputs(self, outputs, name = None) -> Dict[str, Any]:
        idx2outputs = {k['idx']: k['choices'] for k in outputs}
        ord_outputs = [idx2outputs[k['idx']] for k in self.test_ds]

        # parse output text to label id
        def _cut(text):
            return list(jieba.cut(text, cut_all = True))
        id2label = {v:k for k,v in self.label2id.items()}
        preds = parse_bm25_all(ord_outputs, id2label, _cut)
        grounds = [self.label2id[k['charge']] for k in self.test_ds]

        return get_classification_metrics(preds, grounds)