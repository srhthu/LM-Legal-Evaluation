"""
Summarize the case facts
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

@dataclass
class CaseSummary_Config:
    data_path: Optional[str] = None
    task_type: str = 'ljp_summary'
    prompt_file: str = None
    max_len: int = 300

class CaseSummary_Task(TaskBase):
    def __init__(self, config: CaseSummary_Config, tokenizer):
        super().__init__()
        self.config = config

        self.prompt_config = read_json(config.prompt_file)
        self.tokenizer = tokenizer
        self.load_data()
    

    def load_data(self, data_path = None):
        data_path = data_path or self.config.data_path
        data_path = Path(data_path)
        test_ds = read_jsonl(data_path / 'test_data.json')
        self.test_ds = test_ds
    
    