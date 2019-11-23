import re

import pytest
import sly

import mckit.parser.common as cmn
from mckit.parser.common.Lexer import LexerMixin


# @pytest.mark.parametrize("text, expected_types, expected_values", [
#     ("1 0 3.14", ['INTEGER', 'ZERO', 'FLOAT'], [1, 0, 3.14])
# ])
# def test_good_path(text, expected_types, expected_values):
#     lexer = Lexer()
#     tokens = list(t for t in lexer.tokenize(text))
#     result = list(t.type for t in tokens)
#     assert result == expected_types
#     result = list(t.value for t in tokens)
#     assert result == expected_values


# @pytest.mark.parametrize("text, msg_contains", [
#     ("1~ 0 3.14", "column 2"),
#     ("\n1 0 3.14 ~", "at line 2, column 11"),
# ])
# def test_bad_path(text, msg_contains):
#     lexer = Lexer()
#     with pytest.raises(ParseError, match=msg_contains):
#         for _ in lexer.tokenize(text):
#             pass


# noinspection PyUnboundLocalVariable,PyPep8Naming,PyUnresolvedReferences
class MyLexer(sly.Lexer, LexerMixin):
    literals = {':', '(', ')'}
    ignore = ' \t'
    reflags = re.IGNORECASE | re.MULTILINE

    tokens = {NAME, FLOAT, INTEGER, ZERO}

    NAME = r'\d?[A-Za-z-]+'

    @_(cmn.FLOAT)
    def FLOAT(self, token):
        return LexerMixin.on_float(token)

    @_(cmn.INTEGER)
    def INTEGER(self, token):
        res = LexerMixin.on_integer(token)
        if res == 0:
            token.type = 'ZERO'
            token.value = 0
        else:
            token.value = res
        return token

    @_(r'\n+')
    def ignore_newline(self, token):
        self.lineno += len(token.value)

    error = LexerMixin.error


@pytest.mark.parametrize("text, expected_types, expected_values", [
    ("AAA 1 0 3.14", ['NAME', 'INTEGER', 'ZERO', 'FLOAT'], ['AAA', 1, 0, 3.14]),
    ("1B 1 0 3.14", ['NAME', 'INTEGER', 'ZERO', 'FLOAT'], ['1B', 1, 0, 3.14]),
])
def test_good_path(text, expected_types, expected_values):
    lexer = MyLexer()
    tokens = list(lexer.tokenize(text))
    result = list(t.type for t in tokens)
    assert result == expected_types
    result = list(t.value for t in tokens)
    assert result == expected_values


# noinspection PyUnresolvedReferences
class MyParser(sly.Parser):
    tokens = MyLexer.tokens

    @_("NAME number parameters")
    def expression(self, p):
        return p.NAME, p.number, p.parameters

    @_("NAME ZERO")
    def expression(self, p):
        return p.NAME, 0, None

    @_("INTEGER")
    def number(self, p):
        return p.INTEGER

    @_("parameters FLOAT")
    def parameters(self, p):
        res = p.parameters
        res.append(p.FLOAT)
        return res

    @_("FLOAT")
    def parameters(self, p):
        return [p.FLOAT]


@pytest.mark.parametrize("text, expected", [
    ("AAA 1 1.2 3.4", ("AAA", 1, [1.2, 3.4])),
    ("A-Z 0", ("A-Z", 0, None)),
])
def test_parser_with_derived_lexer(text, expected):
    lexer = MyLexer()
    parser = MyParser()
    actual = parser.parse(lexer.tokenize(text))
    assert actual == expected


if __name__ == '__main__':
    pytest.main()
