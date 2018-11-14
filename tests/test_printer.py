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


@pytest.mark.parametrize('value, precision, answer', [
    (5.432e+0, 3, '5.432'),
    (5.432e+1, 3, '54.32'),
    (5.432e+2, 3, '543.2'),
    (5.432e+3, 3, '5432.0'),
    (5.432e+4, 3, '54320.0'),
    (5.432e+5, 3, '543200.0'),
    (5.432e+6, 3, '5432000.0'),
    (5.432e+7, 3, '5.432e+07'),
    (5.432e-1, 3, '0.5432'),
    (5.432e-2, 3, '0.05432'),
    (5.432e-3, 3, '0.005432'),
    (5.432e-4, 3, '0.0005432'),
    (5.432e-5, 3, '5.432e-05'),
    (5.432e-6, 3, '5.432e-06'),
    (1.e-12, 1, '1.0e-12'),
    (1.e-14, 0, '1e-14'),
    (0.0e+0, 1, '0.0'),
    (0.0e+0, 0, '0'),
    (-5.432e+0, 3, '-5.432'),
    (-5.432e+1, 3, '-54.32'),
    (-5.432e+2, 3, '-543.2'),
    (-5.432e+3, 3, '-5432.0'),
    (-5.432e+4, 3, '-54320.0'),
    (-5.432e+5, 3, '-543200.0'),
    (-5.432e+6, 3, '-5432000.0'),
    (-5.432e+7, 3, '-5.432e+07'),
    (-5.432e-1, 3, '-0.5432'),
    (-5.432e-2, 3, '-0.05432'),
    (-5.432e-3, 3, '-0.005432'),
    (-5.432e-4, 3, '-0.0005432'),
    (-5.432e-5, 3, '-5.432e-05'),
    (-5.432e-6, 3, '-5.432e-06'),
    (-1.e-12, 1, '-1.0e-12'),
    (-1.e-14, 0, '-1e-14'),
    (-0.0e+0, 1, '-0.0'),
    (-0.0e+0, 0, '-0')

])
def test_pretty_float(value, precision, answer):
    text = pretty_float(value, precision)
    assert text == answer
