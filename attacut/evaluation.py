from collections import namedtuple

import numpy as np

from nptyping import Array

EvaluationMetrics = namedtuple(
    "EvaluationMetrics",
    ["tp", "fp", "fn", "precision", "recall", "f1"]
)

def compute_metrics(
    labels: Array[np.int32],
    preds: Array[np.int32]
) -> EvaluationMetrics:

    # manually implemented due to keep no. of dependencies minimal
    tp = np.sum(preds * labels)
    fp = np.sum(preds * (1-labels))
    fn = np.sum((1-preds) * labels)

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2 * precision * recall / (precision + recall)

    return EvaluationMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1
    )
