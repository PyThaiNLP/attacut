# -*- coding: utf-8 -*-

import re

import numpy as np
import pytest

from attacut import preprocessing


@pytest.mark.parametrize(
    ("txt", "expected", "steps"),
    [
        (
            "<AZ>This is no. ๑,๒๐๐ </AZ>",
            "This is no. 1,200 ",
            [
                "remove_tags", "thai_digit_to_arabic_digit"
            ]
        ),
        (
            "<AZ>ผมชอบทะเล no ๑,๒๐๐ </AZ>",
            "ผมเกลียดทะเล no 1,200 ",
            preprocessing.DEFAULT_PREPROCESSING_STEPS + [
                lambda x: re.sub(r"ชอบ", "เกลียด", x)
            ]
        )
    ]
)
def test_preprocessing(txt, expected, steps):
    act = preprocessing.preprocess(txt, steps=steps)

    assert act == expected

@pytest.mark.skip(reason="to fix later")
@pytest.mark.parametrize(
    ("txt", "expected", "max_length"),
    [
        (
            "ผม|ไม่|ชอบ|กิน|ผัก| |แต่|นาย| |สมศักษ์ ใจดี| |ชอบ|มากๆ",
            ['ผม|ไม่|ชอบ|กิน|ผัก| |แต่|นาย', ' |สมศักษ์ ใจดี| ', 'ชอบ|มากๆ'],
            8
        ),
        (
            "ผม|ไม่|ชอบ|กิน|ผัก",
            ['ผม|ไม่|ชอบ', 'กิน|ผัก'],
            3
        )
    ]
)
def test_long_txt_sequences(txt, expected, max_length):
    sequences = preprocessing.long_txt_to_sequences(txt, max_length=max_length)

    assert len(sequences) == len(expected)

    for _, s in zip(expected, sequences):
        syllables = preprocessing.syllable_tokenize(s.replace("|", ""))
        assert len(syllables) <= max_length

@pytest.mark.parametrize(
    ("txt", "expected"),
    [
        ("min@adb.com", True),
        ("shoulnotmatch@", False),
        ("something.", False),
        ("min@adb.com|", False)
    ]
)
def test_rx_email(txt, expected):
    is_matched = preprocessing.EMAIL_RX.match(txt)
    assert bool(is_matched) == expected

@pytest.mark.parametrize(
    ("txt", "expected"),
    [
        ("http://soccersuck.com", True),
        ("something", False),
        ("abc.com", True),
        ("www.abc.com", True),
        ("www.080b.com", True),
        ("https://www.080b.com", True),
        ("https:www.080b.com", False)
    ]
)
def test_rx_url(txt, expected):
    is_matched = preprocessing.URL_RX.match(txt)
    assert bool(is_matched) == expected

@pytest.mark.parametrize(
    ("txt", "expected"),
    [
        (
            "IndianInstituteofAdvancedStudies",
            "Indian Instituteof Advanced Studies"
        ),
        (
            "UnitedNation",
            "United Nation"
        ),
        (
            "KOMCHADLUEK",
            "KOMCHADLUEK"
        )
    ]
)
def test_camel_case_expansion(txt, expected):
    actual = preprocessing.expand_camel_case_to_tokens(txt)
    assert " ".join(actual) == expected

@pytest.mark.parametrize(
    ("tokens", "preds", "expected"),
    [ 
        ("acat", "1100", "a|cat"),
        ("ohmygood", "10101000", "oh|my|good")
    ]
)
def test_find_words_from_preds(tokens, preds, expected):

    preds = np.array(list(preds)).astype(int)
    act = preprocessing.find_words_from_preds(tokens, preds)
    exp = expected.split("|")

    assert act == exp


@pytest.mark.parametrize(
    ("txt", "expected"),
    [ 
        ("วันนี้ โรงเรียนเปิด", "วัน~นี้~ ~โรง~เรียน~เปิด"),
        # todo: this case isn't correct, might related to #2
        ("วันนี้   โรงเรียนเปิด", "วัน~นี้~ ~~ ~~ ~โรง~เรียน~เปิด") 
    ]
)
def test_syllable_tokenize(txt, expected):
    act = preprocessing.syllable_tokenize(txt)
    exp = expected.split("~")

    assert act == exp
