import sly
import re
import mckit.surface as surf
import mckit.parser.common as cmn
from .common import drop_c_comments


SURFACE_TYPES = {
    'P', 'PX', 'PY', 'PZ',
    'S', 'SO', 'SX', 'SY', 'SZ',
    'CX', 'CY', 'CZ', 'C/X', 'C/Y', 'C/Z',
    'KX', 'KY', 'KZ', 'K/X', 'K/Y', 'K/Z',
    'TX', 'TY', 'TZ',
    'SQ', 'GQ',
    'X', 'Y', 'Z',
}


def intern_surface_type(word: str):
    word = cmn.ensure_upper(word)
    for w in SURFACE_TYPES:
        if w == word:
            return w
    return None


# noinspection PyPep8Naming,PyUnboundLocalVariable,PyUnresolvedReferences,SpellCheckingInspection
class Lexer(sly.Lexer):
    tokens = {MODIFIER, SURFACE_TYPE, FLOAT, INTEGER}
    ignore = ' \t'
    reflags = re.IGNORECASE | re.MULTILINE

    MODIFIER = r'\*|\+'
    SURFACE_TYPE = r'[a-z]+(?:/[a-z]+)?'
    FLOAT = cmn.FLOAT
    INTEGER = cmn.INTEGER

    @_(r'[a-z]+(?:/[a-z]+)?')
    def SURFACE_TYPE(self, t):
        surface_type = intern_surface_type(t.value)
        if surface_type:
            t.value = surface_type
        else:
            raise cmn.ParseError(f"{t.value} is not a valid surface type")
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

    def __init__(self, transformations, surfaces=dict()):
        sly.Parser.__init__(self)
        self._transformations = transformations
        self._surfaces = surfaces

    @property
    def surfaces(self):
        return self._surfaces

    @_(
        'MODIFIER INTEGER surface_description',
        'INTEGER surface_description',
    )
    def surface(self, p):
        kind, params, transform = p.surface_description
        name = p.INTEGER
        if name in self.surfaces:
            raise cmn.ParseError(f"Surface {name} is duplicated")
        options = {}
        if transform:
            options['transform'] = self.transformations[transform]
        if p.MODIFIER:
            options['modifier'] = p.MODIFIER
        surface = surf.create_surface(kind, *params, **options)
        self.surfaces[name] = surface
        return surface

    @_(
        'INTEGER SURFACE_TYPE surface_params',
        'SURFACE_TYPE surface_params',
    )
    def surface_description(self, p):
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
