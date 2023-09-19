"""
Simple prompts are directly built based on the template and input text.
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
from typing import Any, Dict, Optional, List, Union

from llm_eval.task.base import TaskBase
from llm_eval.utils import read_jsonl, read_json

@dataclass
class SimplePrompt_Config:
    """
    Fill the fields of example into prompts.
    Attributes:
        - fields: field list or fields separated by ','
    """
    data_path: Optional[str] = None
    prompt_template: Optional[str] = None
    task_type: str = 'simple_prompt'
    fields: Union[str, List[str]] = 'text'

    def __post_init__(self):
        if isinstance(self.fields, str):
            self.fields = self.fields.split(',')

class SimplePrompt_Task(TaskBase):
    """
    The prompt is saved in a txt file.
    """
    def __init__(self, config: SimplePrompt_Config, tokenizer = None):
        super().__init__()
        self.config = config

        self.template = open(config.prompt_template, encoding='utf8').read().strip()
        self.tokenizer = tokenizer
        self.data = read_jsonl(config.data_path)
    
    def get_all_subtask(self):
        return ['default']

    def format_template(self, example):
        return self.template.format_map({k:example.get(k, '') for k in self.config.fields})

    def build_subtask(self, name = None):
        subtask = [{'prompt': p} for p in map(self.format_template, self.data)]
        # add the idx field.
        if self.data and 'idx' in self.data[0]:
            subtask = [{'idx': od['idx'], **exa} for exa, od in zip(subtask, self.data)]
        return subtask
    
    def evaluate_outputs(self, outputs, name = None):
        return {}

@dataclass
class SimplePrompt_FieldTrunc_Config(SimplePrompt_Config):
    max_len: int = 300
    task_type: str = 'simple_prompt_ft'

class SimplePrompt_FieldTrunc_Task(SimplePrompt_Task):
    def format_template(self, example):
        """Trunc each field to max_len in config"""
        f_data = {
            k: self.cut_text(example.get(k, ''), self.config.max_len) 
                for k in self.config.fields
        }
        return self.template.format_map(f_data)