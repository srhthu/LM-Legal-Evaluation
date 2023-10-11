"""
The script is started under the dir of the parent dir
"""
import os
import os.path as osp
import sys
sys.path.append(os.path.dirname(sys.path[0]))
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
from pathlib import Path
import glob
import re

from llm_eval.utils import read_json
from llm_eval.evaluator.ppl_mc_eval import MultiChoice_Pipeline

def get_ckpts(path):
    """Find the step of all checkpoint-{step} folder"""
    steps = []
    for name in os.listdir(path):
        m = re.match(r'checkpoint-([0-9]*)', name)
        if m:
            steps.append(m.group(1))
    return steps

def run(model_id, output_dir):
    sub_tasks = 'all'
    run_config = read_json('./config/ljp_mc_ppl.json')
    model_config = run_config['model_config']
    model_config['model_name'] = model_id
    model_config['is_peft'] = True
    model_config['trust_remote_code'] = True

    print('Output path: ', output_dir)
    pipeline = MultiChoice_Pipeline(output_dir, run_config)
    pipeline.do_eval(sub_tasks)

if __name__ == '__main__':
    # pass gpu device in the first argument
    # run from the main dir
    # config here
    run_config = read_json('./config/ljp_mc_ppl.json')
    run_config['model_config']['is_peft'] = True
    run_config['model_config']['trust_remote_code'] = True
    
    pipeline = None
    sub_tasks = 'all'
    for lr in ['1e-5', '2e-5', '5e-6']:
        ft_run_dir = f'/storage/rhshui/workspace/legal-tuning/runs/crimekg/bloomz_7b/lr{lr}_bs16_2gpu'
        model_name = f'crimekg_bloomz_7b/lr{lr}_bs16_2gpu'
        
        steps = get_ckpts(ft_run_dir)

        for step in steps:
            model_id = osp.join(ft_run_dir, f'checkpoint-{step}')
            output_dir = f'./outputs/ljp_mc5_ppl/{model_name}/ckpt-{step}/tpl_1'
            # run(model_id, output_dir)

            print('Output path: ', output_dir)
            
            if pipeline is None:
                # set model_id
                run_config['model_config']['model_name'] = model_id
                pipeline = MultiChoice_Pipeline(output_dir, run_config, late_init = False)
            else:
                pipeline.change_output_dir(output_dir)
                pipeline.load_peft_model(model_id)
            
            pipeline.do_eval(sub_tasks)
        