import re
import sly
from mckit.parser.common import ParseError


# noinspection PyUnresolvedReferences,SpellCheckingInspection
class Lexer(sly.Lexer):
    tokens = {}
    literals = {':', '(', ')'}
    ignore = ' \t'
    reflags = re.IGNORECASE | re.MULTILINE

    def on_float(self, token):
        try:
            token = self.on_integer(token)
        except ValueError:
            token.value = float(token.value)
        return token

    def on_integer(self, token):
        token.value = int(token.value)
        if token.value == 0:
            token.type = 'ZERO'
        else:
            token.type = 'INTEGER'
        return token

    @_(r'\n+')
    def ignore_newline(self, token):
        self.lineno += len(token.value)

    def get_start_of_line(self, token):
        prev_cr = self.text.rfind('\n', 0, token.index)
        # if prev_cr < 0:
        #     prev_cr = 0
        return prev_cr

    def get_end_of_line(self, token):
        next_cr = self.text.find('\n', token.index)
        if next_cr < 0:
            next_cr = len(self.text)
        return next_cr

    @staticmethod
    def column(token, prev_cr):
        return token.index - prev_cr

    def find_column(self, token):
        prev_cr = self.get_start_of_line(token)
        column = Lexer.column(token, prev_cr)
        return column

    def error(self, t):
        prev_cr = self.get_start_of_line(t)
        next_cr = self.get_end_of_line(t)
        column = Lexer.column(t, prev_cr)
        msg = \
            f"Illegal character '{t.value[0]}', at line {self.lineno}, column {self.find_column(t)}\n" + \
            f"{self.text[prev_cr + 1 : next_cr]}\n" + \
            " " * (column - 1) + "^"
        raise ParseError(msg)


# # noinspection PyUnresolvedReferences,SpellCheckingInspection
# class LexerMixin:
#     # tokens = {FLOAT, INTEGER, ZERO}
#     # literals = {':', '(', ')'}
#     # ignore = ' \t'
#     # reflags = re.IGNORECASE | re.MULTILINE
#
#     # FLOAT = cmn.FLOAT
#     # INTEGER = cmn.INTEGER
#     # INTEGER['0'] = ZERO
#
#     # @_(cmn.FLOAT)
#     # def FLOAT(self, token):
#     #     return Lexer.on_float(token)
#     #
#     # @_(cmn.INTEGER)
#     # def INTEGER(self, token):
#     #     return Lexer.on_integer(token)
#
#     # @_(cmn.FLOAT)
#     # def FLOAT(self, token):
#     #     try:
#     #         i_value = int(token.value)
#     #         if i_value == 0:
#     #             token.type = 'ZERO'
#     #         else:
#     #             token.type = 'INTEGER'
#     #         token.value = i_value
#     #     except ValueError:
#     #         token.value = float(token.value)
#     #     return token
#     #
#     # @_(cmn.INTEGER)
#     # def INTEGER(self, token):
#     #     token.value = int(token.value)
#     #     return token
#
#     @staticmethod
#     def on_float(token):
#         try:
#             i_value = int(token.value)
#             if i_value == 0:
#                 token.type = 'ZERO'
#             else:
#                 token.type = 'INTEGER'
#             token.value = i_value
#         except ValueError:
#             token.value = float(token.value)
#         return token
#
#     @staticmethod
#     def on_integer(token):
#         token.value = int(token.value)
#         return token
#
#     # @_(r'\n+')
#     # def ignore_newline(self, token):
#     #     self.lineno += len(token.value)
#
#     def get_start_of_line(self, token):
#         prev_cr = self.text.rfind('\n', 0, token.index)
#         if prev_cr < 0:
#             prev_cr = 0
#         return prev_cr
#
#     def get_end_of_line(self, token):
#         next_cr = self.text.find('\n', token.index)
#         if next_cr < 0:
#             next_cr = len(self.text)
#         return next_cr
#
#     @staticmethod
#     def column(token, prev_cr):
#         return (token.index - prev_cr) + 1
#
#     def find_column(self, token):
#         prev_cr = self.get_start_of_line(token)
#         column = LexerMixin.column(token, prev_cr)
#         return column
#
#     def error(self, t):
#         prev_cr = self.get_start_of_line(t)
#         next_cr = self.get_end_of_line(t)
#         column = LexerMixin.column(t, prev_cr)
#         msg = \
#             f"Illegal character '{t.value[0]}', at line {self.lineno}, column {self.find_column(t)}\n" + \
#             f"{self.text[prev_cr:next_cr]}\n" + \
#             " " * (column - 1) + "^"
#         raise ParseError(msg)
