import re
import string
from typing import Dict, List

import ssg
from attacut.minpythainlp import thai_digit_to_arabic_digit

ARABIC_RX = re.compile(r"[A-Za-z]+")
CAMEL_CASE_RX = re.compile(r"([a-z])([A-Z])([a-z])")
EMAIL_RX = re.compile(r"^\w+\@\w+\.\w+$")
NUMBER_RX = re.compile(r"[0-9,]+")
TRAILING_SPACE_RX = re.compile(r"\n$")
URL_RX = re.compile(r"(https?:\/\/)?(\w+\.)?\w+\.\w+")

DEFAULT_PREPROCESSING_STEPS = [
    "remove_tags", 
    "thai_digit_to_arabic_digit",
    "new_line_as_space",
    "remove_first_pipe",
    "remove_last_pipe"
]


def syllable2token(syllable: str) -> str:
    if ARABIC_RX.match(syllable):
        return "<ENGLISH>"
    elif NUMBER_RX.match(syllable):
        return "<NUMBER>"
    else:
        return syllable


def syllable2ix(sy2ix: Dict[str, int], syllable: str) -> int:
    token = syllable2token(syllable)

    return sy2ix.get(token, sy2ix.get("<UNK>"))


def character2ix(ch2ix: Dict[str, int], character: str) -> int:
    if character == "":
        return ch2ix.get("<PAD>")
    elif character in string.punctuation:
        return ch2ix.get("<PUNC>")

    return ch2ix.get(character, ch2ix.get("<UNK>"))


def step_remove_tags(txt: str) -> str:
    return re.sub(r"<\/?[A-Z]+>", "", txt)


def step_thai_digit_to_arabic_digit(txt: str) -> str:
    return thai_digit_to_arabic_digit(txt)


def step_number_tag(txt: str, tag: str = "ttNumber") -> str:
    return re.sub(r"[0-9,]+", tag, txt)


def step_english_tag(txt: str, tag: str = "ttEnglish") -> str:
    return re.sub(r"[A-Za-z]+", tag, txt)


def step_new_line_as_space(txt: str) -> str:
    return re.sub(r"\n", " ", txt)


def step_remove_first_pipe(txt: str) -> str:
    return re.sub(r"^\|", "", txt)


def step_remove_last_pipe(txt: str) -> str:
    return re.sub(r"\|$", "", txt)


def preprocess(txt: str, steps=DEFAULT_PREPROCESSING_STEPS) -> str:
    for s in steps:
        if isinstance(s, str):
            txt = globals()["step_%s" % s](txt)
        elif callable(s):
            txt = s(txt)
    return txt


def expand_camel_case_to_tokens(w, verbose=0):
    if verbose:
        print(w)

    chars = ""
    for i, c in enumerate(w[1:]):
        ss = w[i-1:i+2]
        if verbose > 3:
            print(ss, chars)
        mm = CAMEL_CASE_RX.match(ss)
        if mm:
            chars = chars + "~" + mm[2]
            if verbose > 3:
                print(">>> match", mm[1], mm[2])
                print("******* ", chars)
        else:
            chars = chars + w[i]
    chars = chars + w[-1]
    return chars.split("~")


def find_words_from_preds(tokens, preds) -> List[str]:
    # Construct words from prediction labels {0, 1}
    curr_word = tokens[0]
    words = []
    for s, p in zip(tokens[1:], preds[1:]):
        if p == 0:
            curr_word = curr_word + s
        else:
            words.append(curr_word)
            curr_word = s

    words.append(curr_word)

    return words


def syllable_tokenize(txt: str) -> List[str]:
    # Proxy function for syllable tokenization, in case we want to try
    # a different syllable tokenizer.
    seps = txt.split(" ")

    new_tokens = []

    for i, s in enumerate(seps):
        tokens = ssg.syllable_tokenize(s)
        new_tokens.extend(tokens)

        if i < len(seps) - 1:
            new_tokens.append(" ")

    return new_tokens
