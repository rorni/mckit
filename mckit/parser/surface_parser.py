from typing import List, Tuple, Optional, NoReturn, Dict, Union, NewType, Callable
import sly
import mckit.surface as surf
from mckit.parser.common import Lexer as LexerBase
import mckit.parser.common.utils as cmn
from mckit.parser.common.utils import drop_c_comments

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


OnAbsentTransformationStrategy = NewType(
    "OnAbsentTransformationStrategy",
    Callable[[int], Optional[surf.Transformation]]
)


class DummyTransformation(surf.Transformation):
    """To substitute transformation when it's not found"""

    def __init__(self, name: int):
        super().__init__(name=name)


def raise_on_absent_transformation_strategy(name: int) -> NoReturn:
    raise KeyError(f"Transformation {name} is not found")


def dummy_on_absent_transformation_strategy(name: int) -> DummyTransformation:
    return DummyTransformation(name)


def ignore_on_absent_transformation_strategy(name: int) -> DummyTransformation:
    return None


# noinspection PyPep8Naming,PyUnboundLocalVariable,PyUnresolvedReferences,SpellCheckingInspection
class Lexer(LexerBase):
    tokens = {MODIFIER, SURFACE_TYPE, FLOAT, INTEGER}

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
        return LexerBase.on_float(t, use_zero=False)

    @_(cmn.INTEGER)
    def INTEGER(self, t):
        return LexerBase.on_integer(t, use_zero=False)


# noinspection PyUnresolvedReferences
class Parser(sly.Parser):
    tokens = Lexer.tokens

    def __init__(
            self,
            transformations: Optional[Dict[int, Union[None, DummyTransformation, surf.Transformation]]] = None,
            on_absent_transformation: Optional[OnAbsentTransformationStrategy] = None,
    ):
        sly.Parser.__init__(self)
        self._transformations = transformations
        if on_absent_transformation is None:
            if transformations:
                self._on_absent_transformation = raise_on_absent_transformation_strategy
            else:
                self._on_absent_transformation = ignore_on_absent_transformation_strategy
        else:
            self._on_absent_transformation = on_absent_transformation

    @property
    def transformations(self):
        return self._transformations

    @property
    def on_absent_transformation(self):
        return self._on_absent_transformation

    def build_surface(
            self,
            name: int,
            kind: str,
            params: List[float],
            transform,
            modifier,
    ) -> surf.Surface:
        options = {'name': name}
        if transform is not None:
            transformation = self.find_transformation(transform)
            if transformation:
                options['transform'] = transformation
        if modifier is not None:
            options['modifier'] = modifier
        _surface = surf.create_surface(kind, *params, **options)
        return _surface

    def find_transformation(self, transform):
        if self.transformations is not None:
            try:
                transformation = self.transformations[transform]
            except KeyError:
                transformation = self.on_absent_transformation(transform)
        else:
            transformation = self.on_absent_transformation(transform)
        return transformation

    @_('MODIFIER  name surface_description')
    def surface(self, p):
        kind, params, transform = p.surface_description  # type: str, List[float], Optional[int]
        return self.build_surface(p.name, kind, params, transform, p.MODIFIER)

    @_('name surface_description')
    def surface(self, p):
        kind, params, transform = p.surface_description  # type: str, List[float], Optional[int]
        return self.build_surface(p.name, kind, params, transform, None)

    @_('INTEGER')
    def name(self, p):
        return p.INTEGER

    @_('transform SURFACE_TYPE surface_params')
    def surface_description(self, p) -> Tuple[str, List[float], Optional[int]]:
        return p.SURFACE_TYPE, p.surface_params, p.transform

    @_(
        'SURFACE_TYPE surface_params',
    )
    def surface_description(self, p) -> Tuple[str, List[float], Optional[int]]:
        return p.SURFACE_TYPE, p.surface_params, None

    @_('INTEGER')
    def transform(self, p):
        return p.INTEGER

    @_('surface_params float')
    def surface_params(self, p) -> List[float]:
        surface_params: List[float] = p.surface_params
        surface_params.append(p.float)
        return surface_params

    @_('float')
    def surface_params(self, p) -> List[float]:
        return [p.float]

    @_('FLOAT')
    def float(self, p):
        return p.FLOAT

    @_('INTEGER')
    def float(self, p) -> float:
        return float(p.INTEGER)


def parse(text, transformations=None, on_absent_transformation=None):
    text = drop_c_comments(text)
    lexer = Lexer()
    parser = Parser(transformations, on_absent_transformation)
    result = parser.parse(lexer.tokenize(text))
    return result
