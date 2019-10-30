# -*- coding: utf-8 -*-

import pytest
from mckit.parser.common import *
import mckit.parser.common as m


@pytest.mark.parametrize("text,expected", [
    (
        """m1
 c bzzz
        1001.21c -1.0
        """,
        """m1
        1001.21c -1.0
        """,
    ),
])
def test_drop_c_comments(text, expected):
    actual = m.drop_c_comments(text)
    assert actual == expected


def test_when_no_c_comments_in_text():
    text = """m1
    1001.21c -1.0
    """
    actual = m.drop_c_comments(text)
    assert actual is text, "drop_c_comments should return the text object without changes"


RE_FLOAT = re.compile(m.FLOAT)


@pytest.mark.parametrize("text,expected", [
    ("1",   "1"),
    (" 0.1", "0.1"),
    (".2\n",  ".2"),
    ("\n -1e10a", "-1e10"),
])
def test_float_pattern(text, expected):
    match = RE_FLOAT.search(text)
    assert match, "Should find float number in the text"
    actual = match.group()
    assert actual == expected
