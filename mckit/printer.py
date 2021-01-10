"""Functions for printing."""
from typing import Any, List, Optional, Text

import warnings

import mckit.constants as constants

from mckit.utils import get_decades, prettify_float, significant_digits

MCNP_FORMATS = {"importance": "{0:.3f}", "material_fraction": "{0:.6e}"}


def print_card(
    tokens: List[Text], offset: int = 8, maxcol: int = 80, sep: str = "\n"
) -> Text:
    """Produce string in MCNP card format.

    Parameters
    ----------
    tokens : list[str]
        List of words to be printed.
    offset : int
        The number of spaces to make continuation of line. Minimum 5.
    maxcol : int
        The maximum length of card line. Maximum 80.
    sep : str
        Separator symbol. This symbol marks positions where newline character
        should be inserted even if maxcol position not reached.

    Returns
    -------
    text : str
        Text string that describes the card.
    """
    if offset < 5:
        offset = 5
        warnings.warn("offset must not be less than 5. offset is set to be 5.")
    if maxcol > 80:
        maxcol = 80
        warnings.warn("maxcol must not be greater than 80. It is set to be 80.")

    length = 0  # current length.
    words = []  # a list of individual words.
    line_sep = "\n" + " " * offset  # separator between lines.
    i = 0
    while i < len(tokens):
        if length + len(tokens[i]) > maxcol or tokens[i] == sep:
            words.append(line_sep)
            length = offset
            while tokens[i] == sep or tokens[i].isspace():
                i += 1
                if i == len(tokens):
                    words.pop()
                    return "".join(words)
        words.append(tokens[i])
        length += len(tokens[i])
        i += 1
    return "".join(words)


def separate(tokens, sep=" "):
    """Adds separation symbols between tokens.

    Parameters
    ----------
    tokens : list[str]
        A list of strings.
    sep : str
        Separator to be inserted between tokens. Default: single space.

    Returns
    -------
    sep_tokens : list
        List of separated tokens.
    """
    sep_tokens = []
    for t in tokens[:-1]:
        sep_tokens.append(t)
        sep_tokens.append(sep)
    sep_tokens.append(tokens[-1])
    return sep_tokens


def print_option(option: str, value: Any) -> List[str]:
    name = option[:3]
    par = option[3:]
    if name == "IMP" and (par == "N" or par == "P" or par == "E"):
        return ["IMP:{0}={1}".format(par, MCNP_FORMATS["importance"].format(value))]
    elif option == "VOL":
        return ["VOL={0}".format(value)]
    elif option == "U":
        return ["U={0}".format(value.name())]
    elif option == "FILL":
        universe = value["universe"]
        tr = value.get("transform", None)
        words = ["FILL={0}".format(universe.name())]
        if tr:
            tr_name = tr.name()
            if not tr_name:
                words[0] = "*" + words[0]
                words.append("(")
                words.extend(tr.get_words())
                words.append(")")
            else:
                words.append("(")
                words.append(str(tr_name))
                words.append(")")
        return words
    else:
        raise ValueError("Incorrect option name: {0}".format(option))


def pretty_float(value, frac_digits: Optional[int] = None) -> str:
    """Pretty print of the float number.

    Parameters
    ----------
    value : float
        Value to be printed.
    frac_digits : int
        The number of digits after decimal point.
    """
    if frac_digits is None:
        frac_digits = significant_digits(
            value, constants.FLOAT_TOLERANCE, resolution=constants.FLOAT_TOLERANCE
        )
    if value == abs(value):
        value = abs(value)
    decades = get_decades(value)
    format_f = "{{0:.{0}f}}".format(max(frac_digits, 0))
    format_e = "{{0:.{0}e}}".format(max(frac_digits + decades, 0))
    text_f = format_f.format(round(value, frac_digits))
    text_e = format_e.format(value)
    if len(text_f) <= len(text_e):
        return text_f
    else:
        return text_e


CELL_OPTION_GROUPS = (
    ("IMPN", "IMPP", "IMPE", "VOL"),  # Importance options
    ("TRCL",),  # Transformation options
    ("U", "FILL"),  # Universe and fill options
)


def add_float(words: List[str], v: float, pretty: bool) -> None:
    words.append(" ")
    if pretty:
        words.append(pretty_float(v))
    else:
        words.append(prettify_float(v))
