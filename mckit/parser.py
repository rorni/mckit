# -*- coding: utf-8 -*-

import re

import ply.lex as lex
import ply.yacc as yacc


literals = ['+', '-', ':', '*', '(', ')', '#', '.']


CELL_KEYWORDS = {
    'IMP', 'VOL', 'PWT', 'EXT', 'FCL', 'WWN', 'DXC', 'NONU', 'PD', 'TMP', 'U',
    'TRCL', 'LAT', 'FILL', 'N', 'P', 'E', 'LIKE', 'BUT', 'RHO', 'MAT'
}

SURFACE_TYPES = {
    'P', 'PX', 'PY', 'PZ', 'S', 'SO', 'SX', 'SY', 'SZ', 'CX', 'CY', 'CZ', 'KX',
    'KY', 'KZ', 'TX', 'TY', 'TZ', 'C/X', 'C/Y', 'C/Z', 'K/X', 'K/Y', 'K/Z',
    'SQ', 'GQ', 'X', 'Y', 'Z'
}

DATA_KEYWORDS = {
    'MODE', 'N', 'P', 'E',
    'VOL', 'AREA', 'TR',
    'IMP', 'ESPLT', 'TSPLT', 'PWT', 'EXT', 'VECT', 'FCL', 'WWE', 'WWN', 'WWP',
    'WWG', 'WWGE', 'PD', 'DXC', 'BBREM',
    'MESH', 'GEOM', 'REF', 'ORIGIN', 'AXS', 'VEC', 'IMESH', 'IINTS', 'JMESH',
    'JINTS', 'KMESH', 'KINTS',
    'SDEF', 'CEL', 'SUR', 'ERG', 'TME', 'DIR', 'VEC', 'NRM', 'POS', 'RAD',
    'EXT', 'AXS', 'X', 'Y', 'Z', 'CCC', 'ARA', 'WGT', 'EFF', 'PAR', 'TR',
    'SI', 'SP', 'SB', 'H', 'L', 'A', 'S', 'D', 'C', 'V', 'DS', 'T','Q', 'SC',
    'SSW', 'SYM', 'PTY', 'SSR', 'OLD', 'NEW', 'COL', 'PSC', 'POA', 'BCW',
    'KCODE', 'KSRC',
    'F', 'FC', 'E', 'T', 'C', 'FQ', 'FM', 'DE', 'DF', 'LOG', 'LIN', 'EM', 'TM',
    'CM', 'CF', 'SF', 'FS', 'SD', 'FU', 'TF', 'DD', 'DXT', 'FT',
    'FMESH', 'GEOM', 'ORIGIN', 'AXS', 'VEC', 'IMESH', 'IINTS', 'JMESH',
    'JINTS', 'KMESH', 'KINTS', 'EMESH', 'EINTS', 'FACTOR', 'OUT', 'TR',
    'M', 'GAS', 'ESTEP', 'NLIB', 'PLIB', 'PNLIB', 'ELIB', 'COND',
    'MPN', 'DRX', 'TOTNU', 'NONU', 'AWTAB', 'XS', 'VOID', 'PIKMT', 'MGOPT',
    'NO',
    'PHYS', 'TMP', 'THTME', 'MT',
    'CUT', 'ELPT', 'NOTRN', 'NPS', 'CTME',
    'PRDMP', 'LOST', 'DBCN', 'FILES', 'PRINT', 'TALNP', 'MPLOT', 'PTRAC',
    'PERT',
    'RAND', 'GEN', 'SEED', 'STRIDE', 'HIST'
}

COMMON_KEYWORDS = {'R', 'I', 'ILOG', 'J', 'NO', 'MESSAGE'}

KEYWORDS = CELL_KEYWORDS.union(DATA_KEYWORDS)

# List of token names
tokens = [
    'blank_line',
    'line_comment',
    'card_comment',
    'continue',
    'separator',
    'surface_type',
    'int_number',
    'flt_number',
    'keyword',
    'title',
    'void_material'
] + list(KEYWORDS)


states = (
    ('continue', 'exclusive'),
    ('cells', 'exclusive'),
    ('ckw', 'exclusive'),
    ('surfs', 'exclusive'),
    ('data', 'exclusive')
)

LINE_COMMENT = r'^[ ]{0,4}C.*'
BLANK_LINE = r'\n(?=[ ]*$)'
CARD_COMMENT = r'\$.*'
CARD_START = r'^[ ]{0,4}[^C\s]'
NEWLINE_SKIP = r'\n(?=' + LINE_COMMENT + r'|[ ]{5,}[^\s])'
RESET_CONTINUE = r'\n(?=[ ]{5,}[^\s])'
CONTINUE = r'&(?=[ ]*(' + CARD_COMMENT + r')?$)'
SEPARATOR = r'\n(?=' + CARD_START + r')'
FRACTION = r'\.'
EXPONENT = r'(E[-+]?\d+)'
INT_NUMBER = r'(\d+)'
FLT_NUMBER = INT_NUMBER + r'?' + FRACTION + INT_NUMBER + EXPONENT + r'?|' +\
             INT_NUMBER + FRACTION + r'?' + EXPONENT + r'|' + \
             INT_NUMBER + FRACTION
KEYWORD = r'[A-Z]+(/[A-Z]+)?'
VOID_MATERIAL = r' 0 '
SKIP = r'[=, ]'

t_ANY_ignore = SKIP


def t_eof(t):
    t.type = 'BLANK_LINE'
    t.value = 'eof'
    return t


def t_title(t):
    r'^.+'
    t.lexer.section_index = 0
    lexer.lineno = 1
    t.lineno = 1
    t.lexer.begin('cells')
    #t.lexer.push_state('continue')
    return t


@lex.TOKEN(BLANK_LINE)
def t_continue_cells_ckw_surfs_data_blank_line(t):
    t.lexer.lineno += 1
    t.lexer.section_index += 1
    if t.lexer.section_index == 1:
        t.lexer.begin('surfs')
    else:
        t.lexer.begin('data')
    t.lexer.push_state('continue')
    return t


@lex.TOKEN(LINE_COMMENT)
def t_continue_cells_ckw_surfs_data_line_comment(t):
    pass


@lex.TOKEN(CARD_COMMENT)
def t_continue_cells_ckw_surfs_data_card_comment(t):
    pass #return t


def t_ANY_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


@lex.TOKEN(CONTINUE)
def t_cells_ckw_surfs_data_continue(t):
    t.lexer.push_state('continue')


@lex.TOKEN(RESET_CONTINUE)
def t_continue_reset_continue(t):
    t.lexer.pop_state()
    t.lexer.lineno += 1


@lex.TOKEN(SEPARATOR)
def t_continue_separator(t):
    t.lexer.lineno += 1
    t.lexer.pop_state()


@lex.TOKEN(SEPARATOR)
def t_ckw_separator(t):
    t.lexer.lineno += 1
    t.lexer.pop_state()
    return t


@lex.TOKEN(SEPARATOR)
def t_INITIAL_surfs_cells_data_separator(t):
    t.lexer.lineno += 1
    return t


@lex.TOKEN(FLT_NUMBER)
def t_cells_ckw_surfs_data_flt_number(t):
    t.value = float(t.value)
    return t


@lex.TOKEN(VOID_MATERIAL)
def t_cells_void_material(t):
    return t


@lex.TOKEN(INT_NUMBER)
def t_cells_ckw_surfs_data_int_number(t):
    t.value = int(t.value)
    return t


@lex.TOKEN(KEYWORD)
def t_cells_ckw_keyword(t):
    value = t.value.upper()
    if value not in CELL_KEYWORDS:
        raise ValueError('Unknown word' + t.value)
    t.type = value
    if t.lexer.current_state() == 'cells':
        t.lexer.push_state('ckw')
    return t


@lex.TOKEN(KEYWORD)
def t_surfs_keyword(t):
    t.value = t.value.upper()
    if t.value not in SURFACE_TYPES:
        raise ValueError('Unknown surface type' + t.value)
    t.type = 'surface_type'
    return t


@lex.TOKEN(KEYWORD)
def t_data_keyword(t):
    value = t.value.upper()
    if value in KEYWORDS:
        t.type = value
    else:
        raise ValueError('Unknown word' + t.value)
    return t


@lex.TOKEN(NEWLINE_SKIP)
def t_continue_cells_ckw_surfs_data_newline_skip(t):
    t.lexer.lineno += 1


lexer = lex.lex(reflags=re.MULTILINE + re.IGNORECASE + re.VERBOSE)


def p_model(p):
    '''model_body : title separator cell_cards blank_line \
                    surface_cards blank_line \
                    data_cards blank_line
    '''
    p[0] = p[1], p[3], p[5], p[7]


def p_cell_cards(p):
    '''cell_cards : cell_cards separator cell_card
                  | cell_card
    '''
    if len(p) == 2:
        # print(p[1])
        name = p[1][0]
        params = p[1][1:]
        p[0] = {name: params}
    elif len(p) == 4:
        # print(p[3])
        name = p[3][0]
        params = p[3][1:]
        p[1][name] = params
        p[0] = p[1]


def p_cell_card(p):
    '''cell_card : int_number cell_material expression cell_options
                 | int_number cell_material expression
                 | int_number LIKE int_number BUT cell_options
    '''
    name = p[1]
    if len(p) == 6:
        params = {'reference': p[3]}
        params.update(p[5])
    else:
        params = {'geometry': p[3]}
        if p[2] is None:
            params['MAT'] = None
        else:
            params['MAT'] = p[2][0]
            params['RHO'] = p[2][1]
        if len(p) == 5:
            params.update(p[4])
        p[0] = name, params


def p_cell_options(p):
    '''cell_options : cell_options cell_option
                    | cell_option'''
    if len(p) == 2:
        p[0] = {p[1][0]: p[1][1]}
    else:
        name, value = p[2]
        p[1][name] = value
        p[0] = p[1]


def p_cell_option(p):
    '''cell_option : fill_option
                   | trcl_option
                   | cell_float_option
                   | cell_int_option
    '''
    p[0] = p[1]


def p_cell_int_option(p):
    '''cell_int_option : U integer
                       | MAT integer
    '''
    p[0] = p[1], p[2]


def p_cell_float_option(p):
    '''cell_float_option : IMP particle_list float
                         | TMP float
                         | RHO float
    '''
    if len(p) == 4:
        name = p[1], *p[2]
        value = p[3]
    else:
        name = p[1]
        value = p[2]
    p[0] = name, value


def p_trcl_option(p):
    '''trcl_option : '*' TRCL '(' param_list ')'
                   | TRCL '(' param_list ')'
                   | TRCL int_number
    '''
    if len(p) == 6:
        indegrees = True
        tr = p[4]
    else:
        indegrees = False
        tr = p[3] if len(p) == 5 else p[2]
    p[0] = 'TRCL', tr, indegrees


def p_fill_option(p):
    '''fill_option : '*' FILL int_number '(' param_list ')'
                   | FILL int_number '(' param_list ')'
                   | FILL int_number
    '''
    if len(p) == 7:
        indegrees = True
        universe = p[3]
        tr = p[5]
    else:
        indegrees = False
        universe = p[2]
        tr = p[4] if len(p) == 6 else None
    p[0] = 'FILL', universe, tr, indegrees


def p_cell_material(p):
    '''cell_material : int_number float
                     | void_material
    '''
    if len(p) == 2:
        p[0] = None
    else:
        p[0] = p[1], p[2]


def p_expression(p):
    '''expression : expression ':' term
                  | term
    '''
    if len(p) == 4:
        p[0] = p[1] + p[3]
        p[0].append('U')
    else:
        p[0] = p[1]


def p_term(p):
    '''term : term factor
            | factor
    '''
    if len(p) == 3:
        p[0] = p[1] + p[2]
        p[0].append('I')
    else:
        p[0] = p[1]


def p_factor(p):
    '''factor : '#' '(' expression ')'
              | '(' expression ')'
              | '-' int_number
              | '+' int_number
              | '#' int_number
              | int_number
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[2]]
        if p[1] == '-':
            p[0].append('C')
        elif p[1] == '#':
            p[0].append('#')
    elif len(p) == 4:
        p[0] = p[2]
    elif len(p) == 5:
        p[0] = p[3] + ['C']


def p_surface_cards(p):
    '''surface_cards : surface_cards separator surface_card
                     | surface_card
    '''
    if len(p) == 2:
        name, *params = p[1]
        p[0] = {name: params}
    elif len(p) == 4:
        print(p[3])
        name, *params = p[3]
        p[1][name] = params
        p[0] = p[1]


def p_surface_card(p):
    '''surface_card : '*' int_number surface_description
                    | '+' int_number surface_description
                    | int_number surface_description
    '''
    if len(p) == 3:
        p[0] = p[1], *p[2], None
    else:
        p[0] = p[2], *p[3], p[1]


def p_surface_description(p):
    '''surface_description : integer surface_type param_list
                           | surface_type param_list'''
    if len(p) == 4:
        p[0] = p[2], p[3], p[1]
    else:
        p[0] = p[1], p[2], None


def p_param_list(p):
    '''param_list : param_list float
                  | float
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + p[2]


def p_float(p):
    '''float : '+' flt_number
             | '-' flt_number
             | flt_number
             | integer
    '''
    if p[1] == '-':
        p[0] = -p[2]
    elif p[1] == '+':
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_integer(p):
    '''integer : '+' int_number
               | '-' int_number
               | int_number
    '''
    if p[1] == '-':
        p[0] = -p[2]
    elif p[1] == '+':
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_data_cards(p):
    '''data_cards : data_cards separator data_card
                  | data_card
    '''
    if len(p) == 2:
        print(p[1])
        name, *params = p[1]
        p[0] = {name: params}
    elif len(p) == 4:
        print(p[3])
        name, *params = p[3]
        p[1][name] = params
        p[0] = p[1]


def p_data_card(p):
    '''data_card : mode_card
                 | material_card
                 | transform_card
    '''
    p[0] = p[1]


def p_mode_card(p):
    '''mode_card : MODE particle_list'''
    p[0] = 'MODE', p[2]


def p_particle_list(p):
    '''particle_list : particle_list particle
                     | particle
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]


def p_particle(p):
    '''particle : N
                | P
                | E
    '''
    p[0] = p[1]


def p_transform_card(p):
    '''transform_card : '*' TR int_number param_list
                      | TR int_number param_list
    '''
    if len(p) == 4:
        shift = 2
        indegrees = False
    else:
        shift = 3
        indegrees = True
    name = p[shift]
    params = p[shift + 1]
    p[0] = 'TR', name, params, indegrees


def p_material_card(p):
    '''material_card : M int_number composition_list material_options
                     | M int_number composition_list
    '''
    if len(p) == 5:
        options = p[4]
    else:
        options = None
    p[0] = 'M', p[2], p[3], options


def p_composition_list(p):
    '''composition_list : composition_list zaid float
                        | zaid float
    '''
    if len(p) == 3:
        p[0] = [(p[1], p[2])]
    else:
        p[0] = p[1] + [(p[2], p[3])]


def p_zaid(p):
    '''zaid : int_number '.' int_number data_spec
            | int_number
    '''
    if len(p) == 5:
        lib = p[3], p[4]
    else:
        lib = None
    p[0] = p[1], lib


def p_data_spec(p):
    '''data_spec : C
                 | D
                 | P
                 | Y
    '''
    p[0] = p[1]


def p_material_options(p):
    '''material_options : material_options mat_opt_name integer
                        | mat_opt_name integer
    '''
    if len(p) == 3:
        p[0] = {p[1]: p[2]}
    else:
        p[1][p[2]] = p[3]
        p[0] = p[1]


def p_mat_opt_name(p):
    '''mat_opt_name : GAS
                    | ESTEP
                    | NLIB
                    | PLIB
                    | PNLIB
                    | ELIB
                    | COND
    '''
    p[0] = p[1]


parser = yacc.yacc()

# with open('..\\tests\\parser_test_data\\lex2.txt') as f:
#     text = f.read()
#     lexer.input(text.upper())
#     while True:
#         tok = lexer.token()
#         print(tok)
#     result = parser.parse(text.upper())
#     print(result)
