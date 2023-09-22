"""
Perplexity based evaluation
"""
import sys,os
from argparse import ArgumentParser

from llm_eval.evaluator.ppl_mc_eval import MultiChoice_Pipeline
from llm_eval.utils import read_jsonl, read_json

def main(args, run_config):
    worker = MultiChoice_Pipeline(args.output_dir, run_config)
    worker.do_eval(args.sub_tasks)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_dir')
    parser.add_argument('--sub_tasks', type = str, help = 'all or subtask names split by ,', default = '5opt-add')
    parser.add_argument('--config', help  = 'config file path.')
    # args to be passed in cmd line
    parser.add_argument('--model_name')
    parser.add_argument('--trust_remote_code', action = 'store_true')

    args = parser.parse_args()

    # For debug
    args.output_dir = 'outputs/debug/ljp_mc_ppl'
    args.sub_tasks = '5opt-add'
    args.config = 'config/ljp_mc_ppl.json'
    args.model_name = '/storage_fast/rhshui/llm/baichuan-7b'
    args.trust_remote_code = True
    print(sys.path)
    print(os.getcwd())
    os.chdir('/storage/rhshui/workspace/legal-eval')

    run_config = read_json(args.config)
    # overwrite some args
    for k in ['model_name', 'trust_remote_code']:
        v = vars(args)[k]
        if v is not None and run_config['model_config'][k] != v:
            run_config['model_config'][k] = v
            print(f'Overwrite config: {k}={v}')

    main(args, run_config)