import typing as tp
from typing import (
    Callable, Iterable, Union, Optional, NoReturn, NewType
)
import sly
from mckit.body import Surface, Shape, Body
from mckit.material import Composition, Material
from mckit.transformation import Transformation
import mckit.parser.common.utils as pu

from mckit.parser.common import Lexer as LexerBase, Index, IgnoringIndex

CELL_WORDS = {
    'U', 'MAT', 'LAT', 'TMP', 'RHO', 'VOL',
}


def intern_cell_word(word: str):
    word = pu.ensure_upper(word)
    word, res = pu.internalize(word, CELL_WORDS)
    if not res:
        raise pu.ParseError(f"'{word}' is not a valid word for cell specification")
    return word


# noinspection PyPep8Naming,PyUnboundLocalVariable,PyUnresolvedReferences,SpellCheckingInspection
class Lexer(LexerBase):
    literals = {':', '(', ')', '*'}
    ignore = '[ \t,=]'
    tokens = {INT_ATTR, IMP, FLOAT_ATTR, TRCL, FILL, INTEGER, FLOAT, ZERO, LIKE, BUT, N, P, E}
    INT_ATTR = 'U|MAT|LAT'
    IMP = 'IMP'
    FLOAT_ATTR = 'TMP|RHO|VOL'
    TRCL = 'TRCL'
    FILL = 'FILL'
    LIKE = 'LIKE'
    BUT = 'BUT'
    N = 'N'
    P = 'P'
    E = 'E'

    @_(pu.FLOAT)
    def FLOAT(self, token):
        return self.on_float(token)

    @_(pu.INTEGER)
    def INTEGER(self, token):
        res = self.on_integer(token)
        if res == 0:
            token.type = 'ZERO'
            token.value = 0
        else:
            token.value = res
        return token


# noinspection PyUnresolvedReferences
class Parser(sly.Parser):
    tokens = Lexer.tokens

    def __init__(
        self,
        surfaces,
        transformations,
        compositions,
        comments,
        trailing_comments,
        on_absent_transformation: Optional[OnAbsentTransformationStrategy] = None,
    ):
        sly.Parser.__init__(self)
        self._surfaces = surfaces
        self._transformations = transformations
        self._compositions = compositions
        self._comments = comments
        self._trailing_comments = trailing_comments
        if on_absent_transformation is None:
            if transformations:
                self._on_absent_transformation = raise_on_absent_transformation_strategy
            else:
                self._on_absent_transformation = ignore_on_absent_transformation_strategy
        else:
            self._on_absent_transformation = on_absent_transformation

    @property
    def surfaces(self):
        return self._surfaces

    @property
    def transformations(self):
        return self._transformations

    @property
    def compositions(self):
        return self._compositions

    @property
    def comments(self):
        return self._comments

    @property
    def traling_comments(self):
        return self._trailing_comments

    def build_cell(self, geometry, options):
        if self.trailing_comments:
            options['comment'] = self.trailing_comments
        return Body(geometry, **options)

    @_('INTEGER cell_material cell_spec')
    def cell(self, p):
        geometry, options = p.cell_spec
        options['name'] = p.INTEGER
        if p.cell_material is not None:
            composition_no, density = p.cell_material  # type: int, float
            comp = self.compositions[composition_no]
            if density > 0:
                material = mat.Material(composition=comp, concentration=density * 1e24)
            else:
                material = mat.Material(composition=comp, density=-density)
            options['MAT'] = material
        return build_cell(geometry, options)

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

    @_('expression : term')
    def expression(self, p):
        return p.expression + p.term

    @_('term')
    def expression(self, p):
        return p.term

    @_('term factor')
    def term(self, p):
        res = p.term + p.factor
        res.append('I')
        return res

    @_('factor')
    def term(self, p):
        return p.factor

    @_('# ( expression )')
    def factor(self, p):
        return p.expression + ['C']

    @_('( expression )')
    def factor(self, p):
        return p.expression

    @_('- integer')
    def factor(self, p):
        surface = self.surfaces[p.integer]
        return [Shape('C', surface)]

    @_('+ integer')
    def factor(self, p):
        surface = self.surfaces[p.integer]
        return [Shape('S', surface)]

    @_('integer')
    def factor(self, p):
        surface = self.surfaces[p.integer]
        return [Shape('S', surface)]

    @_('attributes attribute')
    def attributes(self, p):
        res = p.attributes
        res.update(p.attribute)
        return res

    @_('attribute')
    def attributes(self, p):
        return p

    @_(
        'fill_attribute',
        'trcl_attribute',
        'imp_attribute',
        'float_attribute',
        'int_attribute',
    )
    def attribute(self, p):
        return p[0]

    @_('* FILL integer ( transform_params )')
    def fill_attribute(self, p):
        transform_params = p.transform_params
        transform_params['indegrees'] = True
        fill = {
            'universe': p.integer,
            'transform': Transformation(*p.transform_params)
        }
        return {'FILL': fill}

    @_('FILL integer ( transform_params )')
    def fill_attribute(self, p):
        transform_params = p.transform_params
        transform_params['indegrees'] = False
        fill = {
            'universe': p.integer,
            transform: Transformation(*p.transform_params)
        }
        return {'FILL': fill}

    @_('FILL integer ( integer )')
    def fill_attribute(self, p):
        transform_id: int = p[3]
        transformation = self.transformations[transform_id]
        fill = {
            'universe': p[1],
            transform: transformation
        }
        return {'FILL': fill}

    @_('FILL integer')
    def fill_attribute(self, p):
        fill = {
            'universe': p.integer,
        }
        return {'FILL': fill}

    @staticmethod
    def build_transformation(translation, rotation, in_degrees, inverted):
        return Transformation(
            translation=translation,
            rotation=rotation,
            indegrees=in_degrees,
            inverted=inverted,
        )

    @_('* TRCL ( transform_params )')
    def trcl_attribute(self, p):
        translation, rotation, inverted = p.transform_params
        return {'TRCL': Parser.build_transformation(translation, rotation, True, inverted)}

    @_('TRCL ( transform_params )')
    def trcl_attribute(self, p):
        translation, rotation, inverted = p.transform_params
        return {'TRCL': Parser.build_transformation(translation, rotation, False, inverted)}

    @_('TRCL integer')
    def trcl_attribute(self, p):
        transform_id: int = p[3]
        transformation = self.transformations[transform_id]
        return {'TRCL': transformation}

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

    #
    #  See MCNP, vol1, 3-7
    #  Example: IMP:E,P,N 1 1 0.
    #
    @_('IMP : particle_list float_list')
    def imp_attribute(self, p):
        assert len(p.particle_list) == len(p.float_list), "Lengths of particle and importance values lists differ"
        return dict(('IMP' + k, v) for k, v in zip(p.particle_list, p.float_list))

    @_('float_list float')
    def float_list(self, p):
        return p.float_list + [p.float]

    @_('float')
    def float_list(self, p):
        return [p.float]

    @_('particle_list particle')
    def particle_list(self, p):
        p.particle_list.append(p.particle)
        return p.particle_list

    @_('particle')
    def particle_list(self, p):
        return [p.particle]

    @_('N', 'P', 'E')
    def particle(self, p):
        return pu.ensure_upper(p)

    @_('FLOAT_ATTR float')
    def float_attribute(self, p):
        return {intern_cell_word(p.FLOAT_ATTR): p.float}

    @_('INT_ATTR integer')
    def int_attribute(self, p):
        return {intern_cell_word(p.INT_ATTR): p.integer}

    @_('FLOAT')
    def float(self, p):
        return p.FLOAT

    @_('INTEGER', 'integer')
    def float(self, p):
        return float(p.INTEGER)

    @_('INTEGER', 'ZERO')
    def integer(self, p):
        return p[0]


def parse(
    text: str,
    surfaces: Index = None,
    transformations: Index = None,
    compositions: Index = None
) -> Surface:
    surfaces, transformations, compositions = map(
        lambda x: IgnoringIndex if x is None else x,
        [surfaces, transformations, compositions]
    )
    text = pu.drop_c_comments(text)
    text, comments, trailing_comments = pu.extract_comments(text)
    lexer = Lexer()
    parser = Parser(surfaces, transformations, compositions, comments, trailing_comments)
    result = parser.parse(lexer.tokenize(text))
    return result
