from llm_eval.utils import read_json
from llm_eval.task.ljp import JudgmentPredictionConfig, JudgmentPrediction_Task
from llm_eval.task.base import TaskBase

TASK_MAP = {
    "ljp": (JudgmentPredictionConfig, JudgmentPrediction_Task)
}
class AutoTask:
    @classmethod
    def from_json(cls, path, *args, **kws) -> TaskBase:
        config = read_json(path)
        return cls.from_config(config, *args, **kws)
    
    @classmethod
    def from_config(_, config, *args, **kws) -> TaskBase:
        task_type = config['task_type']
        if task_type not in TASK_MAP:
            raise ValueError(f'task: {task_type} is not implemented')
        
        config_cls, task_cls = TASK_MAP[task_type]
        config = config_cls(**config)
        task = JudgmentPrediction_Task(config, *args, **kws)
        return task