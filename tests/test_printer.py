import pytest

from mckit.printer import *


@pytest.mark.parametrize('tokens, sep, expected', [
    (['1', '2', '3'], ' ', ['1', ' ', '2', ' ', '3']),
    (['1'], ' ', ['1']),
    (['a', 'b', 'c', 'd'], '  ', ['a', '  ', 'b', '  ', 'c', '  ', 'd']),
    (['A', 'B'], '   ', ['A', '   ', 'B'])
])
def test_separate(tokens, sep, expected):
    new_tokens = separate(tokens, sep=sep)
    assert new_tokens == expected


@pytest.mark.parametrize('words, offset, maxcol, sep, expected', [
    (['There', ' ', 'is', '  ', 'a', ' ', 'word', '   ', 'in', '   ', 'here'],
     5, 13, '\n', 'There is  a \n     word   \n     in   \n     here'),
    (['There', ' ', 'is', '  ', 'a', ' ', 'word', '   ', 'in', '   ', 'here'],
     4, 13, '\n','There is  a \n     word   \n     in   \n     here'),
    (['There', ' ', 'is', '  ', 'a', ' ', 'word', '   ', 'in', '   ', 'here'],
     5, 14, '\n', 'There is  a \n     word   in\n     here'),
    (['There', ' ', 'is', '  ', 'a', ' ', 'word', '   ', 'in', '   ', 'here'],
     5, 16, '\n', 'There is  a word\n     in   here'),
    (['There', ' ', 'is', '  ', 'a', ' ', 'word', '   ', 'in', '   ', 'here'],
     6, 16, '\n', 'There is  a word\n      in   here'),
    (['There', ' ', 'is', '  ', 'a', ' ', 'word', '   ', 'in', '   ', 'here'],
     5, 80, '\n', 'There is  a word   in   here'),
    (['There', ' ', 'is', '  ', 'a', '\n', ' ', 'word', '   ', 'in', '   ', 'here'],
     5, 80, '\n', 'There is  a\n     word   in   here'),
    (['There', ' ', 'is', '  ', 'a', ' ', '\n', 'word', '   ', 'in', '   ', 'here'],
     5, 80, '\n', 'There is  a \n     word   in   here'),
    (['There', ' ', 'is', '  ', 'a', '|', ' ', 'word', '   ', 'in', '   ',
      'here'],
     5, 80, '|', 'There is  a\n     word   in   here'),
    (['There', ' ', 'is', '  ', 'a', ' ', '%', 'word', '   ', 'in', '   ',
      'here'],
     5, 80, '%', 'There is  a \n     word   in   here'),
])
def test_print_card(words, offset, maxcol, sep, expected):
    card = print_card(words, offset, maxcol, sep)
    assert card == expected
