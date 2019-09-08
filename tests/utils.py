import pytest

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


def test_add_suffix_to_file_path():
    act = utils.add_suffix_to_file_path("something/input.txt", "omg")
    exp = "something/input-omg.txt"

    assert act == exp


def test_parse_model_params():
    act = utils.parse_model_params("emb:32|l1:48|do:0.5")

    exp = dict(emb=32, l1=48, do=0.5)

    assert act == exp
