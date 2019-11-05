import sly
import re
import mckit.transformation as tr
import mckit.parser.common as cmn
from .common import drop_c_comments


# noinspection PyPep8Naming,PyUnboundLocalVariable,PyUnresolvedReferences,SpellCheckingInspection
class Lexer(sly.Lexer):
    tokens = {NAME, FLOAT, INTEGER}
    literals = {'*', '+'}
    ignore = ' \t'
    reflags = re.IGNORECASE | re.MULTILINE

    NAME = r'[a-z]+(?:/[a-z]+)?'
    FLOAT = cmn.FLOAT
    INTEGER = cmn.INTEGER

    @_(r'[a-z]+(?:/[a-z]+)?')
    def NAME(self, t):
        return t

    @_(cmn.FLOAT)
    def FLOAT(self, t):
        try:
            i_value = int(t.value)
            t.type = 'INTEGER'
            t.value = i_value
        except ValueError:
            t.value = float(t.value)
        return t

    @_(cmn.INTEGER)
    def INTEGER(self, t):
        t.value = int(t.value)
        return t

    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += len(t.value)


# noinspection PyUnresolvedReferences
class Parser(sly.Parser):
    tokens = Lexer.tokens


    @_('* INTEGER NAME surface_params')
    def surface(self, p):
        surf = p.surface_params
        n = len(p)
        surf = p[n - 1]
        surf['name'] = p[n - 2]
        if n == 4:
            surf['modifier'] = p[1]
        comment = extract_comments(p.lineno(1))
        if comment:
            surf['comment'] = comment
        p[0] = surf

    @_('translation rotation INTEGER')
    def transform_params(self, p):
        return p.translation, p.rotation, True

    @_('translation rotation')
    def transform_params(self, p):
        return p.translation, p.rotation, False

    @_('translation')
    def transform_params(self, p):
        return p.translation, None, False

    @_('float float float')
    def translation(self, p):
        return [f for f in p]

    @_(
        'float float float float float float float float float',
        'float float float float float float',
        'float float float float float',
        'float float float',
    )
    def rotation(self, p):
        return [f for f in p]

    @_('FLOAT')
    def float(self, p):
        return p.FLOAT

    @_('INTEGER')
    def float(self, p):
        return float(p.INTEGER)


def parse(text):
    text = drop_c_comments(text)
    lexer = Lexer()
    parser = Parser()
    result = parser.parse(lexer.tokenize(text))
    return result
