# Code below is taken from PyThaiNLP

_thai_arabic = {
    "๐": "0",
    "๑": "1",
    "๒": "2",
    "๓": "3",
    "๔": "4",
    "๕": "5",
    "๖": "6",
    "๗": "7",
    "๘": "8",
    "๙": "9",
}


def thai_digit_to_arabic_digit(text: str) -> str:
    """
    This function convert Thai digits (i.e. ๑, ๓, ๑๐) to Arabic digits
    (i.e. 1, 3, 10).
    :param str text: Text with Thai digits such as '๑', '๒', '๓'
    :return: Text with Thai digits being converted to Arabic digits
             such as '1', '2', '3'
    :rtype: str
    :Example:
    >>> from pythainlp.util import thai_digit_to_arabic_digit
    >>>
    >>> text = 'เป็นจำนวน ๑๒๓,๔๐๐.๒๕ บาท'
    >>> thai_digit_to_arabic_digit(text)
    เป็นจำนวน 123,400.25 บาท
    """
    if not text or not isinstance(text, str):
        return ""

    newtext = []
    for ch in text:
        if ch in _thai_arabic:
            newtext.append(_thai_arabic[ch])
        else:
            newtext.append(ch)

    return "".join(newtext)
