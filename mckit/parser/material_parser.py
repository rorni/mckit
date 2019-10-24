import sly


class Lexer(sly.Lexer):
    tokens = {INTEGER, LIB, FLOAT}
    ignore = ' \t\n'

    LIB = r'\.\d+[cdepuyCDEPUY]'
    INTEGER = r'(\d+)'
    FLOAT = r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'

    @_(r'\d+')
    def INTEGER(self, t):
        t.value = int(t.value)
        return t

    @_(r'\.\d+[cdepuyCDEPUY]')
    def LIB(self, t):
        t.value = t.value[1:].upper()  # drop dot and set upper case
        return t

    @_(r'\d+')
    def FLOAT(self, t):
        t.value = float(t.value)
        return t


# class Parser(sly.Parser):
#     tokens = Lexer.tokens
#     pass