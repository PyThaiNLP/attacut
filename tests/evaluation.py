import numpy as np
import pytest

from attacut import evaluation


@pytest.mark.parametrize(
    ("labels", "preds", "expected"),
    [
        # labels, preds, (tp, fp, fn)
        ( [1, 1, 0], [1, 1, 1], (2, 1, 0) ),
        ( [1, 1, 1], [1, 1, 0], (2, 0, 1) )
    ]
)
def test_something(labels, preds, expected):
    labels = np.array(labels)
    preds = np.array(preds)

    metrics = evaluation.compute_metrics(labels, preds)
    actual = (metrics.tp, metrics.fp, metrics.fn)

    np.testing.assert_array_equal(
        actual,
        expected
    )
