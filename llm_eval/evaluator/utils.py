import os
from pathlib import Path
import json
from tqdm import tqdm
import traceback

def perform_task_timely_save(
    tasks, map_func, save_file, 
    id_field = 'idx', skip_error = True, simple_log = True
):
    """Save task results to a file immediately and skip finished tasks"""
    # Load previous finished tasks if exist
    if Path(save_file).exists():
        prev_task = [json.loads(k) for k in open(save_file, encoding='utf8')]
    else:
        prev_task = []
    prev_idx = set([k[id_field] for k in prev_task])
    print(f'Previous finished: {len(prev_idx)}. Total: {len(tasks)}')
    
    # Filter tasks to complete
    left_tasks = list(filter(lambda k:k[id_field] not in prev_idx, tasks))
    
    # Perform tasks
    for sample in tqdm(left_tasks, ncols = 80):
        try:
            results = map_func(sample)
            # add id field
            if id_field not in results:
                results[id_field] = sample[id_field]
            # write results to a file
            with open(save_file, 'a', encoding='utf8') as f:
                f.write(json.dumps(results, ensure_ascii=False) + '\n')
        except Exception as e:
            if simple_log:
                err_str = str(e)
            else:
                err_str = traceback.format_exc()
                print(sample)
            tqdm.write(f'Error {id_field}={sample[id_field]}, {err_str}')
            if not skip_error:
                exit()