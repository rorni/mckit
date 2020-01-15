import sly
import mckit.transformation as tr
from mckit.parser.common import Lexer as LexerBase
import mckit.parser.common.utils as cmn
from mckit.parser.common.utils import drop_c_comments


# noinspection PyPep8Naming,PyUnboundLocalVariable,PyUnresolvedReferences,SpellCheckingInspection
class Lexer(LexerBase):
    tokens = {NAME, FLOAT, INTEGER}

    NAME = r'\s{0,5}\*?tr\d+'
    FLOAT = cmn.FLOAT
    INTEGER = cmn.INTEGER

    @_(r'\s{0,5}\*?tr\d+')
    def NAME(self, t):
        if t.value[0].isspace():
            t.value = t.value.lstrip()
        if t.value[0] == '*':
            in_degrees = True
            name = int(t.value[3:])
        else:
            in_degrees = False
            name = int(t.value[2:])
        t.value = name, in_degrees
        return t

    @_(cmn.FLOAT)
    def FLOAT(self, t):
        return LexerBase.on_float(t, use_zero=False)

    @_(cmn.INTEGER)
    def INTEGER(self, t):
        return LexerBase.on_integer(t, use_zero=False)


# noinspection PyUnresolvedReferences
class Parser(sly.Parser):
    tokens = Lexer.tokens

    @_('NAME transform_params')
    def transformation(self, p):
        name, in_degrees = p.NAME
        translation, rotation, inverted = p.transform_params
        return tr.Transformation(
            translation=translation,
            rotation=rotation,
            indegrees=in_degrees,
            inverted=inverted,
            name=name,
        )

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
