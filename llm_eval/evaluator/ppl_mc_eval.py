"""
Perplexity Evaluator for Multi-choice questions.
"""
import numpy as np
from llm_eval.ppl import PPL

class MultiChoice_PPL:
    """
    Args:
        metric: how to compare options
            - ppl: averaged negative log probability
            - logp: log probability of the whole option
            - ppl_norm: ppl minus prior ppl of the option
            - logp_norm: logp minus prior logp of the option

    __call__:
        input: dict of fields:
            - question: str
            - options: List[str]
        output: dict of fields:
            - choice(`int`): index of option
            - condition_logp(`List[np.array]`): the conditional log prob of tokens of options,
                i.e., p(option|question)
            - prior_logp(`List[np.array]`): the prior log prob of tokens of options,
                i.e., p(option)
    """
    def __init__(self, model, tokenizer, metric = 'ppl'):
        self.ppl_worker = PPL(model, tokenizer, 'none')
        self.metric = metric
        assert self.metric in ['ppl', 'logp', 'ppl_norm', 'logp_norm']
    
    def __call__(self, example):
        question = example['question']
        options = example['options']

        cond_ppl = [self.ppl_worker.text_ppl(question, k).cpu().tolist() for k in options]
        prior_ppl = [self.ppl_worker.text_ppl(k).cpu().tolist() for k in options]
        
        output_dict = {'cond_ppl': cond_ppl, 'prior_ppl': prior_ppl}
    
        for met in ['ppl', 'logp', 'ppl_norm', 'logp_norm']:
            scores = self.cal_metric(cond_ppl, prior_ppl, met)
            cho = np.argmax(scores)
            output_dict[f'choice_{met}'] = cho

        return output_dict
    
    @staticmethod
    def cal_metric(cond_ppl, prior_ppl, metric):
        """return modified log probability"""
        if metric == 'ppl':
            return [- np.mean(k) for k in cond_ppl]
        elif metric == 'logp':
            return [- np.sum(k) for k in cond_ppl]
        elif metric == 'ppl_norm':
            return [-np.mean(c-p) for c,p in zip(cond_ppl, prior_ppl)]
        elif metric == 'logp_norm':
            return [-np.sum(c-p) for c,p in zip(cond_ppl, prior_ppl)]
        else:
            raise ValueError(f'Error value of metric: {metric}')
    
    
