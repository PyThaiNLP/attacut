import re
import string

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

MAX_SEQUENCE_LENGTH = 64

def map_syllable_token(token):
    if ARABIC_RX.match(token):
        return "<ENGLISH>"
    elif NUMBER_RX.match(token):
        return "<NUMBER>"
    else:
        return token

def mapping_char(ch2ix, c):
    if c == "":
        return ch2ix.get("<PAD>")
    elif c in string.punctuation:
        return ch2ix.get("<PUNC>")
    
    return ch2ix.get(c, ch2ix.get("<UNK>"))

def step_remove_tags(txt):
    return re.sub(r"<\/?[A-Z]+>", "", txt)

def step_thai_digit_to_arabic_digit(txt):
    return thai_digit_to_arabic_digit(txt)

def step_number_tag(txt, tag="ttNumber"):
    return re.sub(r"[0-9,]+", tag, txt)

def step_english_tag(txt, tag="ttEnglish"):
    return re.sub(r"[A-Za-z]+", tag, txt)

def step_new_line_as_space(txt):
    return re.sub(r"\n", " ", txt)

def step_remove_first_pipe(txt):
    return re.sub(r"^\|", "", txt)

def step_remove_last_pipe(txt):
    return re.sub(r"\|$", "", txt)

def preprocess(txt, steps=DEFAULT_PREPROCESSING_STEPS):
    for s in steps:
        if isinstance(s, str):
            txt = globals()["step_%s" % s](txt)
        elif callable(s):
            txt = s(txt)
    return txt

def long_txt_to_sequences(txt, sep="|", max_length=MAX_SEQUENCE_LENGTH):
    tokens = txt.split(sep)

    total_length = 0
    sequences = []
    cur_seq = []

    for t in tokens:
        syllables = syllable_tokenize(t)
        total_syllables = len(syllables)
        new_length = total_length + total_syllables

        if new_length > max_length:
            sequences.append(sep.join(cur_seq))
            total_length, cur_seq = total_syllables, [t]
        else:
            cur_seq.append(t)
            total_length = new_length

    if total_length > 0 and (total_length) < max_length:
        sequences.append(sep.join(cur_seq))
    elif total_length == 0:
        pass
    else:
        raise SystemError(
                "last sequence is longer than %d(%d)\n>%s" % (max_length, len(cur_seq), cur_seq)
            )

    return sequences

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


def find_words_from_preds(tokens, preds):
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


def syllable_tokenize(txt):
    seps = txt.split(" ")

    new_tokens = []

    for i, s in enumerate(seps):
        tokens = ssg.syllable_tokenize(s)
        new_tokens.extend(tokens)

        if i < len(seps) - 1:
            new_tokens.append(' ')

    return new_tokens
