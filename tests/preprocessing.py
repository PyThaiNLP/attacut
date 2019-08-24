# -*- coding: utf-8 -*-

import numpy as np
import pytest
import re

from pythainlp.tokenize import syllable_tokenize

from attacut import pipeline

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
            pipeline.DEFAULT_PREPROCESSING_STEPS + [
                lambda x: re.sub(r"ชอบ", "เกลียด", x)
            ]
        )
    ]
)
def test_preprocessing(txt, expected, steps):
    act = pipeline.preprocess(txt, steps=steps)

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
    sequences = pipeline.long_txt_to_sequences(txt, max_length=max_length)

    assert len(sequences) == len(expected)

    for e, s in zip(expected, sequences):
        syllables = syllable_tokenize(s.replace("|", ""))
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
    is_matched = pipeline.EMAIL_RX.match(txt)
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
    is_matched = pipeline.URL_RX.match(txt)
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
    actual = pipeline.expand_camel_case_to_tokens(txt)
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
    act = pipeline.find_words_from_preds(tokens, preds)
    exp = expected.split("|")

    assert act == exp