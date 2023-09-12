import jieba
from rank_bm25 import BM25Okapi
from typing import Callable, List
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def parse_bm25_all(all_outputs: List[List[str]], id2label, cut_fn: Callable):
    """
    Map the (multiple) output of model to label based on bm25 similarity.
    """
    all_labels = [id2label[i] for i in range(len(id2label))]
    corpus = list(map(cut_fn, all_labels))
    bm25 = BM25Okapi(corpus)

    preds = []
    for outs in all_outputs:
        scores = np.array([
            bm25.get_scores(list(cut_fn(c))) for c in outs
        ]).mean(axis = 0)
        t_id = np.argsort(scores)[-1]
        preds.append(t_id)
    
    return preds

def get_classification_metrics(preds, targets):
    preds = np.array(preds, dtype = np.int64)
    targets = np.array(targets, dtype = np.int64)
    acc = (preds == targets).astype(np.float32).mean()
    p,r,f1, _ = precision_recall_fscore_support(targets, preds, average = 'macro')
    metrics = {'acc': acc,
               'precision': p,
               'recall': r,
               'f1': f1}
    return metrics