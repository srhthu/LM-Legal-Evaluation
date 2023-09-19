"""
High level llm generation agents
"""

import os
import json
from environs import Env
from tqdm import tqdm
import openai
import time
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any
import torch
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import traceback

from transformers import LlamaConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, GenerationConfig

from llm_eval.utils import read_jsonl

PRINT_TRACEBACK = True
@dataclass
class GenerateArguments:
    """Arguments for LM generation. Fields with None value will not be returned"""
    num_output: Optional[int] = None
    max_new_tokens: Optional[int] = None
    # sampling strategy
    do_sample: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    # num_return_sequences: int = 1 # this value will be passed at high level API
    repetition_penalty: Optional[float] = None

    # for openai
    stop: Union[str, List[str]] = '\n'
    # for hf
    decode_save_memory: bool = True 
    # decode one sample each time. Otherwise, decode multiple samples simultaneously

    def to_dict(self):
        return {k: deepcopy(v) for k,v in self.__dict__.items() if v is not None}

@dataclass
class AgentArguments:
    """Arguments to initialize generate agents"""
    agent_type: str # openai, hf
    model: str # model name or path
    
    # HF
    trust_remote_code: bool = False
    torch_dtype: Union[str, torch.dtype] = torch.float16

    def __post_init__(self):
        if not isinstance(self.torch_dtype, torch.dtype):
            if self.torch_dtype == 'float16':
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = 'auto'

class GenerateAgentBase:
    def __init__(
        self, 
        agent_config: AgentArguments, 
        gen_config: GenerateArguments, 
        late_build = False
    ):
        self.gen_config = gen_config
        self.agent_config = agent_config

        self.init()
    
    def init(self):
        raise NotImplementedError
    
    def build_model(self):
        raise NotImplementedError
            
    def generate(self, input_text, num_output = 1):
        """
        Return choices: List[str]
        """
        raise NotImplementedError
    
    def infer_all(
        self, 
        data: List[Dict[str, Any]], 
        save_path: str,
        num_output: Optional[int] = None,
        key_name = 'idx',
        prompt_key = 'prompt'
    ):
        """
        Handle cache and error. Caching enables to resume from previous interrupted job.

        Each data sample should have a key field (default to idx) for caching and a  field indicating the prompt with the name of prompt_key.
        """
        # Read finished samples
        save_path = Path(save_path)
        if save_path.exists():
            prev_out = read_jsonl(save_path)
        else:
            prev_out = []
        prev_idx = set([k[key_name] for k in prev_out])
        print(f'Previous finished: {len(prev_idx)}. Total: {len(data)}')
        # add the key field for identification
        if key_name not in data[0]:
            print(f'Add the index field: {key_name}')
            data = [{key_name: i, **d} for i,d in enumerate(data)]
        # filter unfinished samples
        left_data = list(filter(lambda k:k[key_name] not in prev_idx, data))
        assert len(left_data) == len(data) - len(prev_idx)

        for sample in tqdm(left_data, ncols = 80):
            try:
                choices = self.generate(sample[prompt_key], num_output)
                save_d = {
                    key_name: sample[key_name], 
                    'choices': choices, 
                    prompt_key: sample[prompt_key]
                }
                with open(save_path, 'a', encoding = 'utf8') as f:
                    f.write(json.dumps(save_d, ensure_ascii=False) + '\n')
            except Exception as e:
                if PRINT_TRACEBACK:
                    err = traceback.format_exc()
                    print(f'Error {key_name}={sample[key_name]}, {err}')
                    print(sample)
                    exit()
                else:
                    err_log = f'Error {key_name}={sample[key_name]}, {str(e)}'
                    tqdm.write(err_log)
        


class OpenAI_Agent(GenerateAgentBase):
    def init(self):
        self.model = self.agent_config.model
        
        env = Env()
        env.read_env()
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.endpoint = 'chat' if self.is_chat_endpoint(self.model) else 'complete'
    
    def get_kws(self):
        return dict(
            max_tokens = self.gen_config.max_new_tokens,
            temperature = self.gen_config.temperature,
            stop = self.gen_config.stop
        )

    def complete(self, prompt, n):
        response = openai.Completion.create(
            model = self.model,
            prompt = prompt,
            n = n,
           **self.get_kws()
        )
        choices = [c.text for c in response.choices]
        return choices
    
    def chatcomplete(self, prompt, n):
        """Use the chat endpoint as a completion endpoint"""
        response = openai.ChatCompletion.create(
            model = self.model,
            messages=[{"role": "user", "content": prompt},],
            n = n,
            **self.get_kws()
        )
        choices = [c['message']['content'] for c in response['choices']]
        return choices
    
    def batch_generate(self, input_text_list: List[str], num_output: Optional[int] = None):
        num_output = num_output or self.gen_config.num_output or 1
        assert self.endpoint == 'complete', "Batch generate is only supported by completion endpoint"
        all_choices = self.complete(input_text_list, num_output)
        batch_choices = [all_choices[i*num_output: (i+1)*num_output] for i in range(len(input_text_list))]
        return batch_choices

    def generate(self, input_text, num_output: Optional[int] = None):
        num_output = num_output or self.gen_config.num_output or 1
        if self.endpoint == 'chat':
            choices = self.chatcomplete(input_text, num_output)
        elif self.endpoint == 'complete':
            choices = self.complete(input_text, num_output)
        else:
            raise ValueError(f'endpoint: {self.endpoint}')
        return choices
    
    @staticmethod
    def is_chat_endpoint(model):
        return 'gpt' in model

class Huggingface_Agent(GenerateAgentBase):
    """
    Implement a local LLM with transformers.
    """
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.agent_config.model, 
            trust_remote_code = self.agent_config.trust_remote_code
        )
        late_init = True
        if late_init:
            self.model = None
        else:
            self.build_model()
    
    def get_hf_generation_config(self):
        arg_names = ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'top_k']
        kws = {k:v for k,v in self.gen_config.to_dict().items() if k in arg_names}
        return GenerationConfig(**kws)
    
    def build_model(self):
        config = self.agent_config
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            config.model,
            trust_remote_code = config.trust_remote_code,
            torch_dtype = config.torch_dtype,
            device_map = 'auto'
        )
    
    def generate(self, input_text, num_output: Optional[int] = None):
        if self.model is None:
            self.build_model()
        num_output = num_output or self.gen_config.num_output or 1
        enc = self.tokenizer([input_text], return_tensors = 'pt')
        input_ids = enc.input_ids.cuda()

        hf_gen_cfg = self.get_hf_generation_config()
        
        save_memory = self.gen_config.decode_save_memory
        if save_memory:
            hf_gen_cfg.num_return_sequences = 1
            output_ids = [self.model.generate(input_ids, hf_gen_cfg) for _ in range(num_output)]
            choices = [self.tokenizer.decode(
                k[0, len(input_ids[0]):], skip_special_tokens = True) 
                for k in output_ids]
        else:
            hf_gen_cfg.num_return_sequences = num_output
            output_ids = self.model.generate(input_ids, hf_gen_cfg)
            output_ids = output_ids[:, len(input_ids[0]) :]
            choices = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        return choices


AGENT_MAP = {
    'openai': OpenAI_Agent,
    'hf': Huggingface_Agent
}

class AutoAgent:
    @classmethod
    def from_config(_, agent_config, gen_config) -> GenerateAgentBase:
        cls = AGENT_MAP[agent_config.agent_type](agent_config, gen_config)
        return cls

if __name__ == '__main__':
    # for debug
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('agent_type', help= 'openai or hf')
    parser.add_argument('model', help = 'openai model name or transformers model path')
    args = parser.parse_args()

    agent_args = AgentArguments(args.agent_type, args.model)
    gen_args = GenerateArguments(
        num_output = 3,
        max_new_tokens = 30,
        temperature = 0.8,
        do_sample = True,
        top_p = 1.0,
        top_k = 100,
        repetition_penalty=1.0
    )
    
    # An alternative way to parse argumetns
    # from transformers import HfArgumentParser
    # parser = HfArgumentParser([AgentArguments, GenerateArguments])
    # agent_args, gen_args = parser.parse_args_into_dataclasses()
    
    prompt = 'How to write an essay?'
    print(prompt)

    agent = AutoAgent.from_config(agent_args, gen_args)
    response = agent.generate(prompt)
    print(f'{agent.agent_config.model}:')
    for i, res in enumerate(response):
        print(f'Response {i+1}:\n{res}')
    