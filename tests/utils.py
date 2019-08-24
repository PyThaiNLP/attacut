import pytest
import re

from attacut import utils

@pytest.mark.parametrize(
    ("seq", "expected"),
    [
        (
            (7, 9, 10),
            [
                (0, 7), (7, 16), (16, 26)
            ]
        ),
    ]
)
def test_create_start_stop_indices(seq, expected):
    act = utils.create_start_stop_indices(seq)

    assert act == expected