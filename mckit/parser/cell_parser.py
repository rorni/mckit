import sly
import re
import mckit.surface as surf
import mckit.material as mat
import mckit.parser.common as cmn
from .common import drop_c_comments


# noinspection PyPep8Naming,PyUnboundLocalVariable,PyUnresolvedReferences,SpellCheckingInspection
class Lexer(sly.Lexer):
    tokens = {ATTR, IMP, FLOAT, INTEGER, ZERO}
    literals = {':', '(', ')'}
    ignore = ' \t'
    reflags = re.IGNORECASE | re.MULTILINE

    ATTR=r'U|MAT|LAT|IMP|TMP|RHO|VOL|TRCL|FILL'
    IMP = r'IMP'
    FLOAT = cmn.FLOAT
    INTEGER = cmn.INTEGER

    @_(cmn.FLOAT)
    def FLOAT(self, t):
        try:
            i_value = int(t.value)
            if i_value == t.value:
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

    def __init__(self, transformations, compositions):
        sly.Parser.__init__(self)
        self._transformations = transformations
        self.compositions = compositions

    @property
    def transformations(self):
        return self._transformations

    @property
    def compostions(self):
        return self.compositions

    @_('INTEGER cell_material cell_spec')
    def cell(self, p):
        name = p.INTEGER
        options = {}
        if p.cell_material is not None:
            composition_no, density = p.cell_material  # type: int, float
            comp = self.compositions[composition_no]
            if density > 0:
                material = mat.Material(composition=comp, concentration=density*1e24)
            else:
                material = mat.Material(composition=comp, density=-density)
            options['MAT'] = material
        cell = Body() ...
        return cell

    @_(
        'INTEGER SURFACE_TYPE surface_params',
        'SURFACE_TYPE surface_params',
    )
    def cell_material(self, p):
        params = p.surface_params
        kind = p.SURFACE_TYPE
        transform = p.INTEGER
        return kind, params, transform

    @_('surface_params float')
    def surface_params(self, p):
        return p.surface_params.append(p.float)

    @_('float')
    def surface_params(self, p):
        return [p.float]

    @_('FLOAT')
    def float(self, p):
        return p.FLOAT

    @_('INTEGER')
    def float(self, p):
        return float(p.INTEGER)


def parse(text, transformations={}):
    text = drop_c_comments(text)
    lexer = Lexer()
    parser = Parser(transformations)
    result = parser.parse(lexer.tokenize(text))
    return result
