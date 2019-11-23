import re
import sly
import mckit.material as mat
import mckit.parser.common.utils as cmn
from mckit.parser.common.utils import drop_c_comments
from mckit.parser.common.Lexer import LexerMixin


# noinspection PyPep8Naming,PyUnboundLocalVariable,PyUnresolvedReferences,SpellCheckingInspection
class Lexer(sly.Lexer, LexerMixin):
    literals = {':', '(', ')'}
    ignore = ' \t'
    reflags = re.IGNORECASE | re.MULTILINE
    tokens = {ATTR, IMP, INTEGER, FLOAT, ZERO, LIKE, BUT}
    ATTR = r'U|MAT|LAT|IMP|TMP|RHO|VOL|TRCL|FILL'
    IMP = r'IMP'
    LIKE = r'LIKE'
    BUT = r'BUT'

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


# noinspection PyUnresolvedReferences
class Parser(sly.Parser):
    tokens = Lexer.tokens

    def __init__(self, transformations, compositions):
        sly.Parser.__init__(self)
        self._transformations = transformations
        self._compositions = compositions

    @property
    def transformations(self):
        return self._transformations

    @property
    def compositions(self):
        return self._compositions

    @_('INTEGER cell_material cell_spec')
    def cell(self, p):
        name = p.INTEGER
        options = {}
        if p.cell_material is not None:
            composition_no, density = p.cell_material  # type: int, float
            comp = self.compositions[composition_no]
            if density > 0:
                material = mat.Material(composition=comp, concentration=density * 1e24)
            else:
                material = mat.Material(composition=comp, density=-density)
            options['MAT'] = material
        # cell = Body() ...
        return cell

    @_('INTEGER LIKE INTEGER BUT attributes')
    def cell(self, p):
        raise NotImplementedError()

    @_('INTEGER float')
    def cell_material(self, p):
        return p.INTEGER, p.float

    @_('ZERO')
    def cell_material(self, p):
        return None

    @_('expression attributes')
    def cell_spec(self, p):
        return p.expression, p.attributes

    @_('expression')
    def cell_spec(self, p):
        return p.expression, None

    @_('INTEGER')
    def expression(self, p):
        return p.value

    @_('attributes attribute')
    def attributes(self, p):
        res = p.attributes
        res.append(p.attribute)
        return res

    @_('attribute')
    def attributes(self, p):
        return [p.attribute]

    @_('ATTR')
    def attribute(self, p):
        raise NotImplementedError()

    @_('FLOAT')
    def float(self, p):
        return p.FLOAT

    @_('INTEGER')
    def float(self, p):
        return float(p.INTEGER)


def parse(text, transformations={}):
    raise NotImplementedError()