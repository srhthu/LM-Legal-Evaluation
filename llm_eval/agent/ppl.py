"""
Perplexity Mixin

Support calculation of:
- p(x): the whole ppl of a sentence
- p(x|c), the ppl of the output given the input
"""

from typing import Optional
import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any

from .back import HF_Model_Config, HF_Model_Mixin

class PerplexityMixin:
    def neg_logp(self, input_ids, reduction, start_idx = 0)->torch.Tensor:
        """
        Calculate the log prob from the start_idx token
        Args:
            input_ids: (1, seq_len)
            start_idx: calculate from this token
        """
        input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
        logits = outputs.logits[...,start_idx:-1,:] # (1, seq_len, dim)
        output_ids = input_ids[...,start_idx+1:]
        neg_logp = F.cross_entropy(
            logits.squeeze(0), output_ids.squeeze(0), 
            reduction = reduction
        )
        return neg_logp
    
    def text_neg_logp(self, input_text: str, output_text: str = None, reduction = 'none'):
        """
        If output_text is None, calculate the whole ppl of input_text.
        Otherwise, the whole input is input_text + output_text
        """
        input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids
        if output_text is None:
            start_idx = 0
        else:
            prefix_ids = input_ids
            input_ids = self.tokenizer(
                input_text + output_text, return_tensors='pt'
            ).input_ids
            start_idx = self.max_match_idx(input_ids.squeeze(0), prefix_ids.squeeze(0))
            # start from the last matched token
        return self.neg_logp(input_ids, reduction, start_idx)
    

    def text_ppl(self, input_text, output_text = None):
        return self.text_neg_logp(input_text, output_text, reduction = 'mean')

    def max_match_idx(self, whole, head):
        """Find the index of the last matched token"""
        for idx in range(len(head)-1, 0, -1):
            if head[idx] == whole[idx]:
                return idx
        return idx

class PerplexityAgent(HF_Model_Mixin, PerplexityMixin):
    def __init__(self, model_config: Union[Dict, HF_Model_Config], late_init = False):
        if isinstance(model_config, dict):
            model_config = HF_Model_Config(**model_config)
        self.init_backbone(model_config, late_init)


if __name__ == '__main__':
    ppler = PerplexityAgent({'model_name': 'gpt2'})
    text_1 = 'Hi, this is a demo! '
    text_2 = 'Nice to meet you.'
    print(ppler.tokenizer.tokenize(text_1))
    print(ppler.tokenizer.tokenize(text_1 + text_2))
    
    for reduction in ['none', 'mean']:
        print(f'Reduction = {reduction}')
        ppl1 = ppler.text_neg_logp(text_1, reduction = reduction)
        print(f'ppl 1: {ppl1}')
        ppl2 = ppler.text_neg_logp(text_1, text_2, reduction = reduction)
        print(f'ppl 2: {ppl2}')