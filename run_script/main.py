import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
from pathlib import Path

from llm_eval.utils import read_json
from eval_pipeline import main, Evaluator

MODEL_MAP = {
    'chatglm2_6b': 'THUDM/chatglm2-6b',
    'baichuan_7b': '/storage_fast/rhshui/llm/baichuan-7b',
    'baichuan2_7b': '/storage_fast/rhshui/llm/baichuan2-7b-base',
    'baichuan2_13b': '/storage_fast/rhshui/llm/baichuan2-13b-base',
    'chinese_llama2_7b': '/storage_fast/rhshui/llm/chinese-llama-2-7b',
    'chinese_llama2_13b': '/storage_fast/rhshui/llm/chinese-llama-2-13b',
}

if __name__ == '__main__':
    from argparse import Namespace
    tpl_n = 1
    do_sample = False
    agent_type = 'hf'
    model_name = 'baichuan2_13b'
    save_mem = False

    model = MODEL_MAP[model_name]
    trust = True

    worker = None
    for tpl_n in [4,5]:
        for do_sample in [True, False]:
            output_dir = f'./outputs/ljp_mc5/{model_name}/tpl_{tpl_n}_{do_sample}'
            sub_tasks = 'mc-0-shot',
            pmt_file = Path('/storage/rhshui/workspace/legal-eval/config/template')
            pmt_file = pmt_file / f'ljp_mc_zs_{tpl_n}.json'
            
            run_config = read_json('./config/ljp_mc.json')
            run_config['agent_config']['agent_type'] = agent_type
            run_config['agent_config']['model'] = model
            run_config['agent_config']['trust_remote_code'] = trust
            run_config['task_config']["prompt_config_file"] = str(pmt_file)
            if do_sample:
                run_config['generate_config']['do_sample'] = True
                run_config['generate_config']['temperature'] = 0.8
                run_config['generate_config']['num_output'] = 5
            else:
                run_config['generate_config']['do_sample'] = False
                run_config['generate_config']['num_output'] = 1
            run_config['generate_config']['decode_save_memory'] = save_mem
            
            print('Output path: ', output_dir)
            
            if worker is None:
                worker = Evaluator(output_dir, run_config)
            else:
                worker.set_output_dir(output_dir)
                # worker.set_config(run_config)
                pass
            worker.do_eval(sub_tasks)
            