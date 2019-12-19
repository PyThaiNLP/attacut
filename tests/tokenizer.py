# -*- coding: utf-8 -*-
import pytest

from attacut import SingletonTokenizer, Tokenizer, tokenize


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


def test_singleton_tokenizer():
    tok1 = SingletonTokenizer()
    tok2 = SingletonTokenizer()

    assert tok1 == tok2


def test_tokenize():
    t1 = tokenize("ไปโรงเรียนดีกว่า")
    assert t1 == ["ไป", "โรง", "เรียน", "ดี", "กว่า"]

    t2 = tokenize("ไปด้วยซิ")
    assert t2 == ["ไป", "ด้วย", "ซิ"]

    assert SingletonTokenizer._total_object == 1
