from llm_eval.task.base import TaskBase
from llm_eval.task.ljp import JudgmentPredictionConfig, JudgmentPrediction_Task
from llm_eval.task.ljp_mc import JudgPredMulti_Config, JudgPredMulti_Task


TASK_MAP = {
    "ljp": (JudgmentPredictionConfig, JudgmentPrediction_Task),
    'ljp_mc': (JudgPredMulti_Config, JudgPredMulti_Task)
}
class AutoTask:
    @classmethod
    def from_dict(cls, config_dict, *args, **kws) -> TaskBase:
        task_type = config_dict['task_type']
        if task_type not in TASK_MAP:
            raise ValueError(f'task: {task_type} is not implemented')
        
        config_cls, task_cls = TASK_MAP[task_type]
        config = config_cls(**config_dict)
        task = task_cls(config, *args, **kws)
        return task