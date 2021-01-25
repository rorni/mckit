# -*- coding: utf-8 -*-
from typing import Union

import re

from collections import deque
from pathlib import Path
from warnings import warn

import ply.lex as lex
import ply.yacc as yacc

from mckit.utils.logging import logger

from ..body import Body
from ..material import Composition, Element, Material
from ..surface import create_surface
from ..transformation import Transformation
from ..universe import Universe, produce_universes

warn(
    "The module 'mcnp_input_parser' is deprecated. Use mckit.parser.mcnp_input_sly_parser instead.",
    DeprecationWarning,
)

__DEBUG__ = False

# lex.lex(debug=True, debuglog=log)
# yacc.yacc(debug=True, debuglog=log)

literals = ["+", "-", ":", "*", "(", ")", "#", "."]


CELL_KEYWORDS = {
    "IMP",
    "VOL",
    "PWT",
    "EXT",
    "FCL",
    "WWN",
    "DXC",
    "NONU",
    "PD",
    "TMP",
    "U",
    "TRCL",
    "LAT",
    "FILL",
    "N",
    "P",
    "E",
    "LIKE",
    "BUT",
    "RHO",
    "MAT",
}

SURFACE_TYPES = {
    "P",
    "PX",
    "PY",
    "PZ",
    "S",
    "SO",
    "SX",
    "SY",
    "SZ",
    "CX",
    "CY",
    "CZ",
    "KX",
    "KY",
    "KZ",
    "TX",
    "TY",
    "TZ",
    "C/X",
    "C/Y",
    "C/Z",
    "K/X",
    "K/Y",
    "K/Z",
    "SQ",
    "GQ",
    "X",
    "Y",
    "Z",
    "RPP",
    "BOX",
    "RCC",
}

DATA_KEYWORDS = {
    "MODE",
    "N",
    "P",
    "E",
    "VOL",
    "AREA",
    "TR",
    "IMP",
    "ESPLT",
    "TSPLT",
    "PWT",
    "EXT",
    "VECT",
    "FCL",
    "WWE",
    "WWN",
    "WWP",
    "WWG",
    "WWGE",
    "PD",
    "DXC",
    "BBREM",
    "MESH",
    "GEOM",
    "REF",
    "ORIGIN",
    "AXS",
    "VEC",
    "IMESH",
    "IINTS",
    "JMESH",
    "JINTS",
    "KMESH",
    "KINTS",
    "SDEF",
    "CEL",
    "SUR",
    "ERG",
    "TME",
    "DIR",
    "VEC",
    "NRM",
    "POS",
    "RAD",
    "EXT",
    "AXS",
    "X",
    "Y",
    "Z",
    "CCC",
    "ARA",
    "WGT",
    "EFF",
    "PAR",
    "TR",
    "SI",
    "SP",
    "SB",
    "H",
    "L",
    "A",
    "S",
    "D",
    "C",
    "V",
    "DS",
    "T",
    "Q",
    "SC",
    "SSW",
    "SYM",
    "PTY",
    "SSR",
    "OLD",
    "NEW",
    "COL",
    "PSC",
    "POA",
    "BCW",
    "KCODE",
    "KSRC",
    "F",
    "FC",
    "E",
    "T",
    "C",
    "FQ",
    "FM",
    "DE",
    "DF",
    "LOG",
    "LIN",
    "EM",
    "TM",
    "CM",
    "CF",
    "SF",
    "FS",
    "SD",
    "FU",
    "TF",
    "DD",
    "DXT",
    "FT",
    "FMESH",
    "GEOM",
    "ORIGIN",
    "AXS",
    "VEC",
    "IMESH",
    "IINTS",
    "JMESH",
    "JINTS",
    "KMESH",
    "KINTS",
    "EMESH",
    "EINTS",
    "FACTOR",
    "OUT",
    "TR",
    "M",
    "GAS",
    "ESTEP",
    "NLIB",
    "PLIB",
    "PNLIB",
    "ELIB",
    "COND",
    "MPN",
    "DRX",
    "TOTNU",
    "NONU",
    "AWTAB",
    "XS",
    "VOID",
    "PIKMT",
    "MGOPT",
    "PHYS",
    "TMP",
    "THTME",
    "MT",
    "CUT",
    "ELPT",
    "NOTRN",
    "NPS",
    "CTME",
    "PRDMP",
    "LOST",
    "DBCN",
    "FILES",
    "PRINT",
    "TALNP",
    "MPLOT",
    "PTRAC",
    "PERT",
    "RAND",
    "GEN",
    "SEED",
    "STRIDE",
    "HIST",
}

COMMON_KEYWORDS = {"R", "I", "ILOG", "J", "NO", "MESSAGE"}

KEYWORDS = CELL_KEYWORDS.union(DATA_KEYWORDS)

# List of token names
tokens = (
    [
        "blank_line",
        "line_comment",
        "card_comment",
        "continue",
        "separator",
        "surface_type",
        "int_number",
        "flt_number",
        "keyword",
        "title",
        "void_material",
        "lib_spec",
    ]
    + list(KEYWORDS)
    + list(COMMON_KEYWORDS)
)


states = (
    ("continue", "exclusive"),
    ("cells", "exclusive"),
    ("ckw", "exclusive"),
    ("surfs", "exclusive"),
    ("data", "exclusive"),
)

LINE_COMMENT = r"^[ ]{0,4}C.*"
BLANK_LINE = r"\n(?=[ ]*$)"
CARD_COMMENT = r"\$.*"
CARD_START = r"^[ ]{0,4}[^C\s]"
NEWLINE_SKIP = r"\n(?=" + LINE_COMMENT + r"|[ ]{5,}[^\s])"
RESET_CONTINUE = r"\n(?=[ ]{5,}[^\s])"
CONTINUE = r"&(?=[ ]*(" + CARD_COMMENT + r")?$)"
SEPARATOR = r"\n(?=" + CARD_START + r")"
FRACTION = r"\."
EXPONENT = r"([eE][-+]?\d+)"
INT_NUMBER = r"(\d+)"
FLT_NUMBER = (
    r"("
    + INT_NUMBER
    + r"?"
    + FRACTION
    + INT_NUMBER
    + EXPONENT
    + r"?|"
    + INT_NUMBER
    + FRACTION
    + r"?"
    + EXPONENT
    + r"|"
    + INT_NUMBER
    + FRACTION
    + r")(?=[ \n-+])"
)
# TODO dvp: why FLT_NUMBER is not just: [-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?
LIB_SPEC = INT_NUMBER + r"[CDEPUY]"
KEYWORD = r"[A-Z]+(/[A-Z]+)?"
VOID_MATERIAL = r" 0 "
SKIP = r"[=, ]"

t_ANY_ignore = SKIP

_card_comments = deque()
_line_comments = []


def t_title(t):
    r""".+"""
    t.lexer.section_index = 0
    mcnp_input_lexer.lineno = 1
    t.lineno = 1
    t.lexer.last_pos = 1
    t.lexer.begin("cells")
    # t.mcnp_input_lexer.push_state('continue')
    _line_comments.clear()
    _card_comments.clear()
    return t


@lex.TOKEN(BLANK_LINE)
def t_cells_ckw_surfs_data_blank_line(t):
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos
    t.lexer.section_index += 1
    if t.lexer.section_index == 1:
        t.lexer.begin("surfs")
    else:
        t.lexer.begin("data")
    t.lexer.push_state("continue")
    if _card_comments and _card_comments[-1] is not None:
        _card_comments.append(None)
    return t


@lex.TOKEN(LINE_COMMENT)
def t_continue_cells_ckw_surfs_data_line_comment(t):
    _line_comments.append(t)


@lex.TOKEN(CARD_COMMENT)
def t_continue_cells_ckw_surfs_data_card_comment(t):
    _card_comments.append(t)  # return t


def t_ANY_error(t):
    column = t.lexer.lexpos - t.lexer.last_pos + 1
    msg = r"Illegal character '{0}' at line {1} column {2}".format(
        t.value[0], t.lexer.lineno, column
    )
    raise ValueError(msg, t.value[0], t.lexer.lineno, column)


@lex.TOKEN(CONTINUE)
def t_cells_ckw_surfs_data_continue(t):
    t.lexer.push_state("continue")


@lex.TOKEN(RESET_CONTINUE)
def t_continue_reset_continue(t):
    t.lexer.pop_state()
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos


@lex.TOKEN(SEPARATOR)
def t_continue_separator(t):
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos
    t.lexer.pop_state()


@lex.TOKEN(SEPARATOR)
def t_ckw_separator(t):
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos
    t.lexer.pop_state()
    return t


@lex.TOKEN(SEPARATOR)
def t_INITIAL_surfs_cells_data_separator(t):
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos
    if _card_comments and _card_comments[-1] is not None:
        _card_comments.append(None)
    return t


@lex.TOKEN(FLT_NUMBER)
def t_cells_ckw_surfs_data_flt_number(t):
    t.value = float(t.value)
    return t


@lex.TOKEN(VOID_MATERIAL)
def t_cells_void_material(t):
    return t


@lex.TOKEN(KEYWORD)
def t_cells_ckw_keyword(t):
    value = t.value.upper()
    if value not in CELL_KEYWORDS:
        column = t.lexer.lexpos - t.lexer.last_pos + 1 - len(value)
        msg = r"Unknown keyword '{0}' at line {1} column {2}".format(
            t.value[0], t.lexer.lineno, column
        )
        raise ValueError(msg, t.value[0], t.lexer.lineno, column)
    t.type = value
    t.value = value
    if t.lexer.current_state() == "cells":
        t.lexer.push_state("ckw")
    return t


@lex.TOKEN(KEYWORD)
def t_surfs_keyword(t):
    value = t.value.upper()
    if value not in SURFACE_TYPES:
        column = t.lexer.lexpos - t.lexer.last_pos + 1 - len(value)
        msg = r"Unknown surface type '{0}' at line {1} column {2}".format(
            t.value[0], t.lexer.lineno, column
        )
        raise ValueError(msg, t.value[0], t.lexer.lineno, column)
    t.type = "surface_type"
    t.value = value
    return t


@lex.TOKEN(LIB_SPEC)
def t_data_lib_spec(t):
    t.value = t.value.upper()
    return t


@lex.TOKEN(INT_NUMBER)
def t_cells_ckw_surfs_data_int_number(t):
    t.value = int(t.value)
    return t


@lex.TOKEN(KEYWORD)
def t_data_keyword(t):
    value = t.value.upper()
    if value in KEYWORDS:
        t.type = value
        t.value = value
    else:
        column = t.lexer.lexpos - t.lexer.last_pos + 1 - len(value)
        msg = r"Unknown keyword '{0}' at line {1} column {2}".format(
            t.value[0], t.lexer.lineno, column
        )
        raise ValueError(msg, t.value[0], t.lexer.lineno, column)
    return t


@lex.TOKEN(NEWLINE_SKIP)
def t_continue_cells_ckw_surfs_data_newline_skip(t):
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos


mcnp_input_lexer = lex.lex(
    reflags=re.MULTILINE + re.IGNORECASE + re.VERBOSE,
    debug=__DEBUG__,
    debuglog=logger if __DEBUG__ else None,
)


def extract_comments(line):
    comment = []
    while _card_comments and (
        _card_comments[0] is None or _card_comments[0].lineno < line
    ):
        _card_comments.popleft()
    while _card_comments and _card_comments[0] is not None:
        comment.append(_card_comments.popleft().value.lstrip("$ "))
    return comment


def p_model(p):
    """model_body : title separator cell_cards blank_line \
                    surface_cards blank_line \
                    data_cards blank_line
    """
    p[0] = p[1], p[3], p[5], p[7]


def p_model_without_data(p):
    """model_body : title separator cell_cards blank_line surface_cards blank_line"""
    p[0] = p[1], p[3], p[5], None


def p_cell_cards(p):
    """cell_cards : cell_cards separator cell_card
    | cell_card
    """
    if len(p) == 2:
        p[0] = {p[1]["name"]: p[1]}
    elif len(p) == 4:
        name = p[3]["name"]
        if name in p[1].keys():
            raise ValueError("Duplicate cell name {0}.".format(name))
        p[1][name] = p[3]
        p[0] = p[1]


def p_cell_card(p):
    """cell_card : int_number cell_material expression cell_options
    | int_number cell_material expression
    | int_number LIKE int_number BUT cell_options
    """
    if len(p) == 6:
        params = p[5]
        params["reference"] = p[3]
    else:
        params = p[4] if (len(p) == 5) else {}
        params["geometry"] = p[3]
        if p[2] is not None:
            params["MAT"] = {"composition": p[2][0]}
            if p[2][1] > 0:
                params["MAT"]["concentration"] = p[2][1] * 1.0e24
            else:
                params["MAT"]["density"] = abs(p[2][1])
    params["name"] = p[1]
    comment = extract_comments(p.lineno(1))
    if comment:
        params["comment"] = comment
    p[0] = params


def p_cell_options(p):
    """cell_options : cell_options cell_option
    | cell_option"""
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[1].update(p[2])
        p[0] = p[1]


def p_cell_option(p):
    """cell_option : fill_option
    | trcl_option
    | cell_float_option
    | cell_int_option
    """
    p[0] = p[1]


def p_cell_int_option(p):
    """cell_int_option : U integer
    | MAT integer
    | LAT integer
    """
    p[0] = {p[1]: p[2]}


def p_cell_float_option(p):
    """cell_float_option : IMP ':' particle_list float
    | TMP float
    | RHO float
    | VOL float
    """
    if len(p) == 5:
        p[0] = {p[1] + par: p[4] for par in p[3]}
    else:
        p[0] = {p[1]: p[2]}


def p_trcl_option(p):
    """trcl_option : '*' TRCL '(' transform_params ')'
    | TRCL '(' transform_params ')'
    | TRCL int_number
    """
    if len(p) == 3:
        p[0] = {"TRCL": p[2]}
    elif len(p) == 5:
        p[0] = {"TRCL": p[3]}
    else:
        p[4]["indegrees"] = True
        p[0] = {"TRCL": p[4]}


def p_fill_option(p):
    """fill_option : '*' FILL int_number '(' transform_params ')'
    | FILL int_number '(' transform_params ')'
    | FILL int_number '(' int_number ')'
    | FILL int_number
    """
    if len(p) == 7:
        p[5]["indegrees"] = True
        p[0] = {"FILL": {"universe": p[3], "transform": p[5]}}
    elif len(p) == 6:
        p[0] = {"FILL": {"universe": p[2], "transform": p[4]}}
    else:
        p[0] = {"FILL": {"universe": p[2]}}


def p_cell_material(p):
    """cell_material : int_number float
    | void_material
    """
    if len(p) == 2:
        p[0] = None
    else:
        p[0] = p[1], p[2]


def p_expression(p):
    """expression : expression ':' term
    | term
    """
    if len(p) == 4:
        p[0] = p[1] + p[3]
        p[0].append("U")
    else:
        p[0] = p[1]


def p_term(p):
    """term : term factor
    | factor
    """
    if len(p) == 3:
        p[0] = p[1] + p[2]
        p[0].append("I")
    else:
        p[0] = p[1]


def p_factor(p):
    """factor : '#' '(' expression ')'
    | '(' expression ')'
    | '-' int_number
    | '+' int_number
    | '#' int_number
    | int_number
    """
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[2]]
        if p[1] == "-":
            p[0].append("C")
        elif p[1] == "#":
            p[0].append("#")
    elif len(p) == 4:
        p[0] = p[2]
    elif len(p) == 5:
        p[0] = p[3] + ["C"]


def p_surface_cards(p):
    """surface_cards : surface_cards separator surface_card
    | surface_card
    """
    if len(p) == 2:
        p[0] = {p[1]["name"]: p[1]}
    elif len(p) == 4:
        name = p[3]["name"]
        if name in p[1].keys():
            raise ValueError("Duplicate surface name {0}".format(name))
        p[1][name] = p[3]
        p[0] = p[1]


def p_surface_card(p):
    """surface_card : '*' int_number surface_description
    | '+' int_number surface_description
    | int_number surface_description
    """
    n = len(p)
    surf = p[n - 1]
    surf["name"] = p[n - 2]
    if n == 4:
        surf["modifier"] = p[1]
    comment = extract_comments(p.lineno(1))
    if comment:
        surf["comment"] = comment
    p[0] = surf


def p_surface_description(p):
    """surface_description : integer surface_type param_list
    | surface_type param_list"""
    n = len(p)
    descr = {"params": p[n - 1], "kind": p[n - 2]}
    if n == 4:
        descr["transform"] = p[1]
    p[0] = descr


def p_param_list(p):
    """param_list : param_list float
    | float
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]


def p_float(p):
    """float : '+' flt_number
    | '-' flt_number
    | flt_number
    | integer
    """
    if p[1] == "-":
        p[0] = -p[2]
    elif p[1] == "+":
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_integer(p):
    """integer : '+' int_number
    | '-' int_number
    | int_number
    """
    if p[1] == "-":
        p[0] = -p[2]
    elif p[1] == "+":
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_data_cards(p):
    """data_cards : data_cards separator data_card
    | data_card
    """
    if len(p) == 2:
        name, value = p[1]
        p[0] = {name: value}
    elif len(p) == 4:
        name, value = p[3]
        if name in p[1].keys() and isinstance(value, dict):
            p[1][name].update(value)
        else:
            p[1][name] = value
        p[0] = p[1]


def p_data_card(p):
    """data_card : mode_card
    | material_card
    | transform_card
    """
    p[0] = p[1]


def p_mode_card(p):
    """mode_card : MODE particle_list"""
    p[0] = "MODE", p[2]


def p_particle_list(p):
    """particle_list : particle_list particle
    | particle
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]


def p_particle(p):
    """particle : N
    | P
    | E
    """
    p[0] = p[1]


def p_transform_card(p):
    """transform_card : '*' TR int_number transform_params
    | TR int_number transform_params
    """
    n = len(p)
    tr = p[n - 1]
    tr["name"] = p[n - 2]
    if n == 5:
        tr["indegrees"] = True
    comment = extract_comments(p.lineno(1))
    if comment:
        tr["comment"] = comment
    p[0] = "TR", {tr["name"]: tr}


def p_transform_params(p):
    """transform_params : translation rotation integer
    | translation rotation
    | translation
    """
    tr = {"translation": p[1]}
    if len(p) > 2:
        tr["rotation"] = p[2]
    if len(p) == 4:
        tr["inverted"] = True
    p[0] = tr


def p_translation(p):
    """translation : float float float"""
    p[0] = [p[1], p[2], p[3]]


def p_rotation(p):
    """rotation : float float float float float float float float float
    | float float float float float float
    | float float float float float
    | float float float
    """
    p[0] = [p[i] for i in range(1, len(p))]


def p_material_card(p):
    """material_card : M int_number composition_list material_options
    | M int_number composition_list
    """
    if len(p) == 5:
        p[3].update(p[4])
    p[3]["name"] = p[2]
    comment = extract_comments(p.lineno(1))
    if comment:
        p[3]["comment"] = comment
    p[0] = "M", {p[2]: p[3]}


def p_composition_list(p):
    """composition_list : composition_list zaid_fraction
    | zaid_fraction
    """
    if len(p) == 2:
        key = p[1][0]
        value = p[1][1:]
        p[0] = {key: [value]}
    else:
        key = p[2][0]
        value = p[2][1:]
        if key in p[1].keys():
            p[1][key].append(value)
        else:
            p[1][key] = [value]
        p[0] = p[1]


def p_zaid_fraction(p):
    """zaid_fraction : int_number '.' lib_spec float
    | int_number float
    """
    n = len(p)
    key = "atomic" if p[n - 1] > 0 else "weight"
    opts = {}
    if n == 5:
        opts["lib"] = p[3]
    p[0] = key, p[1], opts, abs(p[n - 1])


def p_material_options(p):
    """material_options : material_options material_option
    | material_option
    """
    if len(p) == 2:
        p[0] = {p[1][0]: p[1][1]}
    else:
        p[1][p[2][0]] = p[2][1]
        p[0] = p[1]


def p_material_option(p):
    """material_option : GAS integer
    | ESTEP integer
    | NLIB lib_spec
    | PLIB lib_spec
    | PNLIB lib_spec
    | ELIB lib_spec
    | COND integer
    """
    p[0] = p[1], p[2]


mcnp_input_parser = yacc.yacc(
    tabmodule="mcnp_input_tab",
    debug=__DEBUG__,
    debuglog=logger,
    errorlog=logger if __DEBUG__ else yacc.NullLogger(),
)


def read_mcnp(filename: Union[str, Path], encoding: str = "cp1251") -> Universe:
    warn(
        "The function 'read_mcnp' is deprecated. Use mckit.parser.from_file() instead.",
        DeprecationWarning,
    )
    with open(filename, encoding=encoding) as f:
        text = f.read()
    return read_mcnp_text(text)


def read_mcnp_text(
    text: str,
) -> Universe:
    warn(
        "The function 'read_mcnp_text' is deprecated. Use mckit.parser.from_text() instead.",
        DeprecationWarning,
    )
    text = (
        text.rstrip() + "\n"
    )  # float number spec requires end of line or space after float
    mcnp_input_lexer.begin("INITIAL")
    title, cells, surfaces, data = mcnp_input_parser.parse(
        text, tracking=True, lexer=mcnp_input_lexer, debug=__DEBUG__
    )
    bodies = []
    for name in list(cells.keys()):
        bodies.append(_get_cell(name, cells, surfaces, data))
    return produce_universes(bodies)


def _get_transformation(name, data):
    if isinstance(name, dict):
        return Transformation(**name)
    assert isinstance(name, int)
    tr = data["TR"][name]
    if isinstance(tr, dict):
        tr = Transformation(**tr)
        data["TR"][name] = tr
    return tr


def _get_composition(name, data):
    comp = data["M"][name]
    if isinstance(comp, dict):
        for i, (el, opt, frac) in enumerate(comp.get("atomic", [])):
            comp["atomic"][i] = (Element(el, **opt), frac)
        for i, (el, opt, frac) in enumerate(comp.get("weight", [])):
            comp["weight"][i] = (Element(el, **opt), frac)
        comp = Composition(**comp)
        data["M"][name] = comp
    return comp


def _get_surface(name, surfaces, data):
    surf_data = surfaces[name]
    if isinstance(surf_data, dict):
        kind = surf_data.pop("kind")
        params = surf_data.pop("params")
        tr = surf_data.get("transform", None)
        if isinstance(tr, int):
            surf_data["transform"] = _get_transformation(tr, data)
        surf = create_surface(kind, *params, **surf_data)
        surfaces[name] = surf
    else:
        surf = surf_data
    return surf


def _get_cell(name, cells, surfaces, data):
    if isinstance(cells[name], Body):  # Already exists
        return cells[name]
    cell_data = cells.pop(name)  # To avoid circular references and infinite
    # cycle
    geometry = cell_data.pop("geometry", None)
    if geometry is None:  # find reference cell
        ref_name = cell_data.pop("reference")
        ref_cell = _get_cell(ref_name, cells, surfaces, data)
        geometry = ref_cell.shape
        for k, v in ref_cell.options.items():
            if k not in cell_data.keys():
                cell_data[k] = v
        rho = cell_data.pop("RHO", None)
        material = cell_data["MAT"]
        cell_data["MAT"] = {"composition": material.composition.name()}
        if rho is not None:
            if rho > 0:
                cell_data["MAT"]["concentration"] = rho * 1.0e24
            else:
                cell_data["MAT"]["density"] = abs(rho)
        else:
            cell_data["MAT"]["density"] = material.density
    else:  # create geometry from polish notation
        for i, g in enumerate(geometry):
            if isinstance(g, int):
                if i + 1 < len(geometry) and geometry[i + 1] == "#":
                    comp_cell = _get_cell(g, cells, surfaces, data)
                    shape = comp_cell.shape
                    tr = comp_cell.options.get("TRCL", None)
                    if tr:
                        shape = shape.transform(tr)
                    geometry[i] = shape
                    geometry[i + 1] = "C"
                else:
                    geometry[i] = _get_surface(g, surfaces, data)

    # Create material if necessary
    mat_data = cell_data.get("MAT", None)
    if mat_data:
        comp = mat_data["composition"]
        mat_data["composition"] = _get_composition(comp, data)
        cell_data["MAT"] = Material(**mat_data)

    # Replace transformations
    if "TRCL" in cell_data.keys():
        cell_data["TRCL"] = _get_transformation(cell_data["TRCL"], data)
    fill = cell_data.get("FILL", {})
    temp_tr = None
    if "transform" in fill.keys():
        temp_tr = fill["transform"]
        temp_tr = _get_transformation(temp_tr, data)
        if temp_tr and not isinstance(temp_tr, Transformation):
            print(name)
        fill["transform"] = temp_tr

    cell = Body(geometry, **cell_data)
    cells[name] = cell
    return cell
