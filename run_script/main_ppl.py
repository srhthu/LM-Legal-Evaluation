"""
The script is started under the dir of the parent dir
"""
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
from pathlib import Path

from llm_eval.utils import read_json
from llm_eval.evaluator.ppl_mc_eval import MultiChoice_Pipeline

MODEL_MAP = {
    'chatglm2_6b': 'THUDM/chatglm2-6b',
    'baichuan_7b': '/storage_fast/rhshui/llm/baichuan-7b',
    'baichuan2_7b': '/storage_fast/rhshui/llm/baichuan2-7b-base',
    'baichuan2_13b': '/storage_fast/rhshui/llm/baichuan2-13b-base',
    'chinese_llama2_7b': '/storage_fast/rhshui/llm/chinese-llama-2-7b',
    'chinese_llama2_13b': '/storage_fast/rhshui/llm/chinese-llama-2-13b',
    'bloomz_7b': '/storage_fast/rhshui/llm/bloomz-7b1-mt',
}

if __name__ == '__main__':
    model_name = 'baichuan2_7b' if len(sys.argv) < 3 else sys.argv[2]
    save_mem = False

    model = MODEL_MAP[model_name]
    trust = True

    output_dir = f'./outputs/ljp_mc5_ppl/{model_name}/tpl_1'
    sub_tasks = 'all'
    pmt_file = Path('/storage/rhshui/workspace/legal-eval/config/template') / f'ljp_mc_ppl_1.json'

    run_config = read_json('./config/ljp_mc_ppl.json')
    model_config = run_config['model_config']
    model_config['model_name'] = model
    model_config['trust_remote_code'] = trust

    print('Output path: ', output_dir)
    pipeline = MultiChoice_Pipeline(output_dir, run_config)
    pipeline.do_eval(sub_tasks)