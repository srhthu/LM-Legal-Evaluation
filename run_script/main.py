import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
from pathlib import Path

from llm_eval.utils import read_json
from eval_pipeline import main

if __name__ == '__main__':
    from argparse import Namespace
    tpl_n = 1
    do_sample = False
    agent_type = 'hf'
    # model = '/storage_fast/rhshui/llm/baichuan-7b'
    model = 'THUDM/chatglm2-6b'
    trust = True

    for do_sample in [True, False]:
        output_dir = f'./outputs/ljp_mc5/chatglm2_6b/tpl_{tpl_n}_{do_sample}'
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
        

        main(output_dir, sub_tasks, run_config)