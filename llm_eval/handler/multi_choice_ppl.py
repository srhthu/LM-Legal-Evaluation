"""
Task handler of perplexity evaluation of multi-choice questions
"""
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any
import torch
import numpy as np

from llm_eval.agent import HF_Model_Mixin, PerplexityMixin, HF_Model_Config, PerplexityAgent



class MultiChoicePPL(PerplexityAgent):
    """
    Given a question and options, predict the option with lowest perplexity:
        metric: how to calculate perplexity
            - ppl: averaged negative log probability
            - neg_logp: negative log probability of the whole option
            - ppl_norm: ppl minus prior ppl of the option
            - neg_logp_norm: negative logp minus prior logp of the option
    """
    def __call__(self, example) -> Dict[str, Union[int, List[int]]]:
        """
        Args: dict of fields:
            - question: str
            - options: List[str]
        Return: dict of fields:
            - cond_nlogp(`List[List[float]]`): conditional negtive log likelihood
            - prior_nlogp(`List[List[float]]`): prior negtive log likelihood
            - choice_{metric}(`int`): predicted option index of the {metric}
            - scores_{metric}(`List[float]`): option scores of the {metric}
        """
        question = example['question']
        options = example['options']
        cond_nlogp = [self.text_neg_logp(
            question, k, reduction = 'none').cpu().numpy() 
            for k in options
        ]
        prior_nlogp = [self.text_neg_logp(k, reduction = 'none').cpu().numpy() for k in options]

        output_dict = {'cond_nlogp': [k.tolist() for k in cond_nlogp], 'prior_nlogp': [k.tolist() for k in prior_nlogp]}
        # output_dict = {}
        for met in ['ppl', 'neg_logp', 'ppl_norm', 'neg_logp_norm']:
            scores = self.cal_metric(cond_nlogp, prior_nlogp, met)
            output_dict[f'scores_{met}'] = np.array(scores).tolist()
            cho = np.argmin(scores).tolist()
            output_dict[f'choice_{met}'] = cho

        return output_dict
    
    def cal_metric(
        self, 
        cond_neg_logp: List[np.ndarray], 
        prior_neg_logp: List[np.ndarray], 
        metric
    )->np.ndarray:
        if metric == 'ppl':
            return [self.mean(k) for k in cond_neg_logp]
        elif metric == 'neg_logp':
            return [np.sum(k) for k in cond_neg_logp]
        elif metric == 'ppl_norm':
            return [self.mean(c)-self.mean(p) for c,p in zip(cond_neg_logp, prior_neg_logp)]
        elif metric == 'neg_logp_norm':
            return [np.sum(c)-np.sum(p) for c,p in zip(cond_neg_logp, prior_neg_logp)]
        else:
            raise ValueError(f'Error value of metric: {metric}')
    
    def mean(self, x: List[float]):
        # deal with empty list
        if len(x) == 0:
            return 0
        return np.mean(x)

    @property
    def metrics(self):
        return ['ppl', 'neg_logp', 'ppl_norm', 'neg_logp_norm']