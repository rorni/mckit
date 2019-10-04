# -*- coding: utf-8 -*-

import pytest
from io import StringIO

from mckit.parser.mcnp_section_parser import *


@pytest.mark.parametrize("text,expected", [
    (
            "1 0 1\n",
            ('1 0 1\n', 0)
    ),
    (
            "1 0 1 $bla bla bla\n",
            ('1 0 1\n', 1)
    ),
    (
            """
1 0 1 $bla bla bla
c the comment
   2 -3
"""[1:-1],
            (
                    """
1 0 1
   2 -3
"""[1:-1], 2)
    ),
])
def test_comment_pattern(text, expected):
    actual = COMMENT_PATTERN.subn('', text)
    assert expected == actual


@pytest.mark.parametrize("text,comment,card", [
    (
            "1 0 1\n",
            None,
            '1 0 1\n'
    ),
    (
            """
c the preceding comment
1 0 1
"""[1:-1],
            """
c the preceding comment
"""[1:],
            """
1 0 1
"""[1:-1]
    ),
    (
            """
c the preceding comment
1 0 1
c inner comment
  2 $continuation
"""[1:-1],
            """
c the preceding comment
"""[1:],
            """
1 0 1
c inner comment
  2 $continuation
"""[1:-1]
    ),
])
def test_card_pattern(text, comment, card):
    res = CARD_PATTERN.match(text)
    groups = res.groupdict()
    actual_comment = groups["comment"]
    assert comment == actual_comment
    actual_card = groups["card"]
    assert card == actual_card


@pytest.mark.parametrize("text,cards,kind", [
    (
            """
c the preceding comment
1 0 1
c inner comment
  2 $continuation
"""[1:-1],
            [
                Card("c the preceding comment"),
                Card(
                    """
1 0 1
c inner comment
  2 $continuation
"""[1:-1], kind=Kind.CELL)
            ],
            Kind.CELL,
    ),
    (
            """
c the preceding comment
1 0 1
c inner comment
  2 $continuation
c the second preceding comment
2 0 -1 $the next card
c the trailing comment
c the second line of the trailing comment
"""[1:-1],
        [
            Card("c the preceding comment"),
            Card(
                """
1 0 1
c inner comment
  2 $continuation
"""[1:-1], kind=Kind.CELL),
            Card("c the second preceding comment"),
            Card("2 0 -1 $the next card", kind=Kind.CELL),
            Card(
                    """
c the trailing comment
c the second line of the trailing comment
"""[1:-1]),
            ],
            Kind.CELL,
    ),
])
def test_split_to_cards(text, cards, kind):
    actual_cards = list(split_to_cards(text, kind))
    assert actual_cards == cards


def test_card_constructor():
    descr = "1 0 1"
    c = Card(descr)
    assert c.text == descr


def test_input_sections_constructor():
    title = "Testing"
    cell_cards = [Card("1 0 1"), Card("2 0 -1")]
    surface_cards = [Card("1 so 100")]
    t = InputSections(
        title,
        cell_cards,
        surface_cards,
        [Card("sdef")],
    )
    assert t.title == title


@pytest.mark.parametrize("text,expected,kind", [
    (
            """
1 0 1 $bla bla bla
"""[1:-1],
            [
                Card("1 0 1", kind=Kind.CELL)
            ],
            Kind.CELL,
    ),
    (
            """
c some comment
c second line
1 0 1 $bla bla bla
   2 $continuation
c trailing comment
"""[1:-1],
            [
                Card(
                    """
1 0 1 2
"""[1:-1], kind=Kind.CELL
                )
            ],
            Kind.CELL,
    ),
    (
            """
c some comment
c second line
1 0 1 $bla bla bla
   2 $continuation
c trailing comment
2
  0 -1  $rrrrrr
c z-z-zz-z-z-z
"""[1:-1],
            [
                Card(
                    """
1 0 1 2
"""[1:-1], kind=Kind.CELL
                ),
                Card(
                    """
2 0 -1
"""[1:-1], kind=Kind.CELL
                ),
            ],
            Kind.CELL,
    ),
])
def test_clean_mcnp_cards(text, expected, kind):
    actual = list(clean_mcnp_cards(split_to_cards(text, kind)))
    assert expected == actual


@pytest.mark.parametrize("text,expected", [
    (
            """
test
1 0 1 $bla bla bla

1 so 1

sdef
"""[1:],
            None,
    ),
    (
            """
test
c some comment
c second line
1 0 1 $bla bla bla
   2 $continuation
2 0 -1
c trailing comment

1 so 1

sdef
"""[1:],
            None,
    ),
    (
            """
test
c some comment
c second line
1 0 1 $bla bla bla
   2 $continuation
c trailing comment
2
  0 -1  $rrrrrr
c z-z-zz-z-z-z

1 so 1

sdef
"""[1:],
            None
    ),
    (
            """
continue
ctme 3000
"""[1:],
            None
    ),
])
def test_print(text, expected):
    stream = StringIO(text)
    sections = parse_sections(stream)
    out = StringIO()
    sections.print(out)
    if expected is None:
        expected = text.strip()
    actual = out.getvalue().strip()
    assert actual == expected
