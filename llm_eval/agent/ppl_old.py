"""
Utilities about perplexity.

Support calculate of:
- p(x)
- p(x|c), c is the context
"""
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer,T5ForConditionalGeneration, LlamaForCausalLM

class PPL:
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        reduction = 'none'
    ):
        assert reduction in ['none', 'mean']

        self.model = model
        self.tokenizer = tokenizer
        self.reduction = reduction

        self.ce = torch.nn.CrossEntropyLoss(reduction = 'none')
    
    def _log_prob(self, input_ids, start_idx = 0)->torch.Tensor:
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
        cross_entropy = F.cross_entropy(
            logits.squeeze(0), output_ids.squeeze(0), 
            reduction = self.reduction
        )
        return -cross_entropy
    
    def text_ppl(self, input_text: str, output_text: str = None):
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
        return -self._log_prob(input_ids, start_idx)

    def token_prob(self, input_text:str, output_text: str = None):
        ppl = self.text_ppl(input_text, output_text)
        return torch.exp(-ppl)

    def max_match_idx(self, whole, head):
        """Find the index of the last matched token"""
        for idx in range(len(head)-1, 0, -1):
            if head[idx] == whole[idx]:
                return idx
        return idx

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    # model.to('cuda:0')
    tk = AutoTokenizer.from_pretrained('gpt2')

    ppler = PPL(model, tk, reduction = 'none')
    text_1 = 'Hi, this is a demo! '
    text_2 = 'Nice to meet you.'
    print(tk.tokenize(text_1))
    print(tk.tokenize(text_1 + text_2))
    
    for reduction in ['none', 'mean']:
        ppler.reduction = reduction
        print(f'Reduction = {reduction}')
        ppl1 = ppler.text_ppl(text_1)
        print(f'ppl 1: {ppl1}')
        prob1 = ppler.token_prob(text_1)
        print(f'prob 1: {prob1}')
        ppl2 = ppler.text_ppl(text_1, text_2)
        print(f'ppl 2: {ppl2}')
        prob2 = ppler.token_prob(text_1, text_2)
        print(f'prob 2: {prob2}')