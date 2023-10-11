"""
Implement the backbone
"""
import os
from environs import Env
import openai
import torch
from transformers import LlamaConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, GenerationConfig
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

class OpenAI_Mixin:
    def init_backbone(self):
        self.set_api_key()
    
    def set_api_key(self):
        env = Env()
        env.read_env()
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    def complete(self, prompt, n) -> List[str]:
        response = openai.Completion.create(
            model = self.model,
            prompt = prompt,
            n = n,
           **self.get_kws()
        )
        choices = [c.text for c in response.choices]
        return choices
    
    def chatcomplete(self, prompt, n) -> List[str]:
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

@dataclass
class HF_Model_Config:
    model_name: str = None
    is_peft: bool = False
    trust_remote_code: bool = False
    torch_dtype: Union[str, torch.dtype] = torch.float16
    device_map: str = 'auto'

    def __post_init__(self):
        if not isinstance(self.torch_dtype, torch.dtype):
            if self.torch_dtype == 'float16':
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = 'auto'

class HF_Model_Mixin:
    """
    Model initialization and late init.

    Args:
        model_name
        kws: keywords to initialize hf models, including
            - trust_remote_code(`bool`)
            - torch_dtype(`str` or torch.dtype)
            - device_map(`str` or `dict`)

    Attributes:
        - model
        - tokenizer
    """
    def init_backbone(self, model_config: HF_Model_Config, late_init = False):
        self._model_config = model_config
        self._late_init = late_init

        self._base_model = model_config.model_name
        if model_config.is_peft:
            peft_config = PeftConfig.from_pretrained(model_config.model_name)
            self._base_model = peft_config.base_model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._base_model, 
            trust_remote_code = model_config.trust_remote_code
        )
        if late_init:
            self._model: PreTrainedModel = None
        else:
            self._init_model()
    
    def _init_model(self):
        config = self._model_config
        kws = self.get_init_kws()
        if not config.is_peft:
            self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                config.model_name, **kws
            )
        else:
            self._model: PeftModel = AutoPeftModelForCausalLM.from_pretrained(
                config.model_name, **kws
            )
    
    def get_init_kws(self):
        config = self._model_config
        kws = dict(
            trust_remote_code = config.trust_remote_code,
            torch_dtype = config.torch_dtype,
            device_map = config.device_map
        )
        return kws
           
    def load_peft_model(self, model_id):
        """This should be called after initialization"""
        if self._model is None:
            self._init_model()
        
        assert isinstance(self._model, PeftModel)
        base_model = self._model.get_base_model()
        kws = self.get_init_kws()
        # load peft parameters
        self._model = PeftModel.from_pretrained(base_model, model_id, **kws)
        print('Reload peft model: ', model_id)
        # rename model config
        self._model_config.model_name = model_id
    
    @property
    def model(self):
        if self._model is None:
            self._init_model()
        return self._model
