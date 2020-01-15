import re
import sly
import mckit.material as mat
from mckit.parser.common.utils import drop_c_comments, extract_comments
import mckit.parser.common.utils as cmn
from mckit.parser.common.Lexer import Lexer as LexerBase


# noinspection PyPep8Naming,PyUnboundLocalVariable,PyUnresolvedReferences,SpellCheckingInspection
class Lexer(LexerBase):
    tokens = {NAME, FRACTION, OPTION, FLOAT}

    OPTION = r'(?:(?:gas|estep|cond)\s+\d+)|(?:(?:n|p|pn|e)lib\s+\S+)'

    @_(r'm\d+')
    def NAME(self, t):
        t.value = int(t.value[1:])
        return t

    @_(r'\d+(?:\.\d+[cdepuy])')
    def FRACTION(self, t):
        if '.' in t.value:
            isotope, lib = t.value.split('.')
            isotope = int(isotope)
            lib = lib[:] # drop dot
            lib = cmn.ensure_lower(lib)
            t.value = isotope, lib
        else:
            t = self.on_integer(t)
        return t

    @_(cmn.FLOAT)
    def FLOAT(self, token):
        return LexerBase.on_float(token)


# noinspection PyUnresolvedReferences
class Parser(sly.Parser):
    tokens = Lexer.tokens

    def __init__(self, comments, trailing_comments):
        self.comments = comments
        self.trailing_comments = trailing_comments

    def build_composition(self, name, fractions, options=None):
        atomic = []
        weight = []

        for el, fraction in fractions:
            if fraction < 0.0:
                weight.append((el, -fraction))
            else:
                atomic.append((el, fraction))

        if options is None:
            options = {}

        options['name'] = name

        if self.trailing_comments:
            options['comment'] = self.trailing_comments

        return mat.Composition(atomic=atomic, weight=weight, **options)

    @_('composition_a')
    def composition(self, p):
        name, fractions, options = p.composition_a
        return self.build_composition(name, fractions, options)

    @_('NAME fractions options')
    def composition_a(self, p):
        return p.NAME, p.fractions, p.options

    @_('NAME fractions')
    def composition_a(self, p):
        return p.NAME, p.fractions, None

    @_('fractions fraction')
    def fractions(self, p):
        p.fractions.append(p.fraction)
        return p.fractions

    @_('fraction')
    def fractions(self, p):
        return [p.fraction]

    # @_('fraction_a EOL_COMMENT')
    # def fraction(self, p):
    #     name, lib, frac = p.fraction_a
    #     return mat.Element(name, lib=lib, comment=p.EOL_COMMENT), frac
    #
    @_('fraction_a')
    def fraction(self, p):
        name, lib, frac = p.fraction_a
        return mat.Element(name, lib=lib), frac

    @_('FRACTION FLOAT')
    def fraction_a(self, p):
        isotope, lib = p.FRACTION
        return isotope, lib, p.FLOAT

    @_('options option')
    def options(self, p):
        option, value = p.option
        p.options[option] = value
        return p.options

    @_('option')
    def options(self, p):
        option, value = p.option
        result = dict()
        result[option] = value
        return result

    @_('OPTION')
    def option(self, p):
        option, value = p.OPTION.split()
        if not option.islower():
            option = option.lower()
        if option in ("gas", "estep", "cond"):
            value = int(value)
        return option, value


def parse(text):
    text = drop_c_comments(text)
    text, comments, trailing_comments = extract_comments(text)
    lexer = Lexer()
    parser = Parser(comments, trailing_comments)
    result = parser.parse(lexer.tokenize(text))
    return result
