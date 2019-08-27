import pytest
import re

from attacut import Tokenizer

@pytest.mark.parametrize(
    ("model", "txt", "expected"),
    [
        ("attacut-sc", "ภาษาไทยยากจัง", "ภาษา|ไทย|ยาก|จัง"),
        ("attacut-c", "ภาษาไทยยากจัง", "ภาษา|ไทย|ยาก|จัง"),
    ]
)
def test_tokenization(model, txt, expected):
    atta = Tokenizer(model)
    act = atta.tokenize(txt)

    assert act == expected.split("|")