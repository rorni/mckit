# -*- coding: utf-8 -*-

import re

import ply.lex as lex
import ply.yacc as yacc


literals = ['+', '-', ':', '*', '(', ')', '#', ',', '.']


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
] + list(CELL_KEYWORDS.union(DATA_KEYWORDS))


states = (
    ('continue', 'inclusive'),
    ('title', 'exclusive'),
    ('cells', 'inclusive'),
    ('ckw', 'inclusive'),
    ('surfs', 'inclusive'),
    ('data', 'inclusive')
)

LINE_COMMENT = r'^[ ]{0,4}C.*'
BLANK_LINE = r'\n[ ]*$'
CARD_COMMENT = r'\$.*'
CARD_START = r'^[ ]{0,4}[^C\s]'
SEPARATOR = r'\n(?=(' + LINE_COMMENT + r'\n)*' + CARD_START + r')'
CONTINUE = r'&'
RESET_CONTINUE = r'\n(?=' + CARD_START + r')'
MANTISSA = r'(\.\d+)'
EXPONENT = r'(E[-+]?\d+)'
INT_NUMBER = r'(\d+)'
FLT_NUMBER = INT_NUMBER + r'?' + MANTISSA + EXPONENT + r'?|' + INT_NUMBER + EXPONENT
KEYWORD = r'[A-Z]+(/[A-Z]+)?'
VOID_MATERIAL = r' 0 '
SKIP = r'[= ]'

t_ANY_ignore = SKIP


def t_eof(t):
    t.type = 'BLANK_LINE'
    t.value = 'eof'
    return t


def t_title_title(t):
    r'^.+'
    t.lexer.begin('cells')
    t.lexer.push_state('continue')
    return t


@lex.TOKEN(BLANK_LINE)
def t_ANY_blank_line(t):
    t.lexer.lineno += 1
    if t.lexer.states:
        section_state = t.lexer.states.pop()
    else:
        section_state = 'INITIAL'
    t.lexer.begin(section_state)
    t.lexer.push_state('continue')
    return t


@lex.TOKEN(LINE_COMMENT)
def t_ANY_line_comment(t):
    pass


@lex.TOKEN(CARD_COMMENT)
def t_ANY_card_comment(t):
    pass #return t


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


@lex.TOKEN(RESET_CONTINUE)
def t_continue_reset(t):
    t.lexer.lineno += 1
    t.lexer.pop_state()


@lex.TOKEN(CONTINUE)
def t_continue(t):
    t.lexer.push_state('continue')


@lex.TOKEN(SEPARATOR)
def t_ckw_separator(t):
    t.lexer.lineno += 1
    t.lexer.pop_state()
    return t


@lex.TOKEN(SEPARATOR)
def t_INITIAL_surfs_cells_title_separator(t):
    t.lexer.lineno += 1
    return t


@lex.TOKEN(FLT_NUMBER)
def t_flt_number(t):
    t.value = float(t.value)
    return t


@lex.TOKEN(VOID_MATERIAL)
def t_cells_void_material(t):
    return t


@lex.TOKEN(INT_NUMBER)
def t_int_number(t):
    t.value = int(t.value)
    return t


@lex.TOKEN(KEYWORD)
def t_cells_keyword(t):
    if t.value not in CELL_KEYWORDS:
        raise ValueError('Unknown word' + t.value)
    t.type = t.value
    t.lexer.push_state('ckw')
    return t


@lex.TOKEN(KEYWORD)
def t_surfs_keyword(t):
    if t.value not in SURFACE_TYPES:
        raise ValueError('Unknown surface type' + t.value)
    t.type = 'surface_type'
    return t


@lex.TOKEN(KEYWORD)
def t_data_keyword(t):
    if t.value in DATA_KEYWORDS:
        t.type = t.value
    else:
        raise ValueError('Unknown word' + t.value)
    return t


def t_ANY_newline_skip(t):
    r'\n'
    t.lexer.lineno += 1


lexer = lex.lex(reflags=re.MULTILINE + re.IGNORECASE + re.VERBOSE)
lexer.begin('title')
lexer.states = ['data', 'surfs']


#def p_model(p):
#    '''model : message_block BLANK_LINE model_body
#             | model_body
#    '''
#    pass


def p_model_body(p):
    '''model_body : TITLE NEWLINE cell_cards BLANK_LINE \
                    surface_cards BLANK_LINE \
                    data_cards
    '''
    p[0] = p[1], p[3], p[5], p[7]


def p_cell_cards(p):
    '''cell_cards : cell_cards cell_card
                  | cell_card
    '''
    if len(p) == 2:
        print(p[1])
        name = p[1][0]
        params = p[1][1:]
        p[0] = {name: params}
    elif len(p) == 3:
        print(p[2])
        name = p[2][0]
        params = p[2][1:]
        p[1][name] = params
        p[0] = p[1]


def p_cell_card(p):
    '''cell_card : INT_NUMBER cell_material expression cell_options NEWLINE
                 | INT_NUMBER cell_material expression NEWLINE
                 | INT_NUMBER LIKE INT_NUMBER BUT cell_options NEWLINE'''
    name = p[1]
    if len(p) == 7:
        params = {'REFERENCE': p[3]}
        params.update(p[5])
    else:
        params = {'GEOMETRY': p[3]}
        if p[2] is None:
            params['MAT'] = None
        else:
            params['MAT'] = p[2][0]
            params['RHO'] = p[2][1]
        if len(p) == 6:
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
    '''cell_option : VOL float
                   | TMP float
                   | U INT_NUMBER
                   | importance_option
                   | fill_option
    '''
    if len(p) == 3:
        p[0] = p[1], p[2]
    else:
        p[0] = p[1]


def p_importance_option(p):
    '''importance_option : IMP ':' particle float'''
    p[0] = 'IMP', (p[3], p[4])


def p_fill_option(p):
    '''fill_option : FILL INT_NUMBER
                   | FILL INT_NUMBER '(' INT_NUMBER ')'
                   | FILL INT_NUMBER '(' param_list ')'
                   | '*' FILL INT_NUMBER '(' param_list ')'
    '''
    if len(p) == 3:
        p[0] = 'FILL', p[2],
    elif len(p) == 6:
        p[0] = 'FILL', p[2], p[4]
    elif len(p) == 7:
        p[0] = 'FILL', p[3], p[5], True


def p_particle(p):
    '''particle : N
                | P
                | E'''
    p[0] = p[1]


def p_cell_material(p):
    '''cell_material : INT_ZERO
                     | INT_NUMBER float'''
    if len(p) == 2:
        p[0] = None
    else:
        p[0] = p[1], p[2]


def p_expression(p):
    '''expression : expression ':' term
                  | term'''
    if len(p) == 4:
        p[0] = p[1] + p[3]
        p[0].append('U')
    else:
        p[0] = p[1]


def p_term(p):
    '''term : term factor
            | factor'''
    if len(p) == 3:
        p[0] = p[1] + p[2]
        p[0].append('I')
    else:
        p[0] = p[1]


def p_factor(p):
    '''factor : '#' '(' expression ')'
              | '(' expression ')'
              | '-' INT_NUMBER
              | '+' INT_NUMBER
              | '#' INT_NUMBER
              | INT_NUMBER
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
    '''surface_cards : surface_cards surface_card
                     | surface_card
    '''
    if len(p) == 2:
        name, *params = p[1]
        p[0] = {name: params}
    elif len(p) == 3:
        print(p[2])
        name, *params = p[2]
        p[1][name] = params
        p[0] = p[1]


def p_data_cards(p):
    '''data_cards : data_cards data_card
                  | data_card
    '''
    if len(p) == 2:
        print(p[1])
        name, *params = p[1]
        p[0] = {name: params}
    elif len(p) == 3:
        print(p[2])
        name, *params = p[2]
        p[1][name] = params
        p[0] = p[1]


def p_data_card(p):
    '''data_card : mode_card
                 | volume_card
    '''
    print('fff')
    p[0] = p[1]


def p_mode_card(p):
    '''mode_card : MODE particle_list NEWLINE'''
    p[0] = 'MODE', p[2]
  #  print(p[2])


def p_particle_list(p):
    '''particle_list : particle_list particle
                     | particle
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]


def p_volume_card(p):
    '''volume_card : VOL integer_list NEWLINE
                   | VOL NO integer_list NEWLINE
    '''
    if len(p) == 4:
        p[0] = 'VOL', p[2]
    else:
        p[0] = 'VOL', p[3]


def p_integer_list(p):
    '''integer_list : integer_list INT_NUMBER
                    | INT_NUMBER
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]


def p_transform_card(p):
    '''transform_card : '*' TR INT_NUMBER param_list
                      | TR INT_NUMBER param_list
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


# def p_material_card(p):
#     '''material_card : M INT_NUMBER composition_list material_options'''
#     pass
#
#
# def p_composition_list(p):
#     '''composition_list : composition_list zaid float
#                         | zaid float'''
#     pass
#
# def p_zaid(p):
#     '''zaid : INT_NUMBER '.' INT_NUMBER'''
#     pass


def p_surface_card(p):
    '''surface_card : '*' INT_NUMBER surface_description NEWLINE
                    | '+' INT_NUMBER surface_description NEWLINE
                    | INT_NUMBER surface_description NEWLINE
    '''
    if len(p) == 4:
        p[0] = p[1], *p[2], None
    else:
        p[0] = p[2], *p[3], p[1]


def p_surface_description(p):
    '''surface_description : integer surface_spec param_list
                           | surface_spec param_list'''
    if len(p) == 4:
        p[0] = (p[2], p[3], p[1])
    else:
        p[0] = (p[1], p[2], None)


def p_surface_spec(p):
    '''surface_spec : X
                    | Y
                    | Z
                    | P
                    | PX
                    | PY
                    | PZ
                    | S
                    | SO
                    | SX
                    | SY
                    | SZ
                    | CX
                    | CY
                    | CZ
                    | C_X
                    | C_Y
                    | C_Z
                    | KX
                    | KY
                    | KZ
                    | K_X
                    | K_Y
                    | K_Z
                    | TX
                    | TY
                    | TZ
                    | SQ
                    | GQ'''
    p[0] = p[1]


def p_integer(p):
    '''integer : '+' INT_NUMBER
               | '-' INT_NUMBER
               | '+' INT_ZERO
               | '-' INT_ZERO
               | INT_NUMBER
               | INT_ZERO
    '''
    if p[1] == '-':
        p[0] = -p[2]
    elif p[1] == '+':
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_param_list(p):
    '''param_list : param_list float
                  | param_list skip_list
                  | repeat_list
                  | multiply_list
                  | interpolate_list
                  | skip_list
                  | float
    '''
    if len(p) == 2:
        if not isinstance(p[1], list):
            p[0] = [p[1]]
        else:
            p[0] = p[1]
    else:
        if not isinstance(p[2], list):
            p[0] = p[1] + [p[2]]
        else:
            p[0] = p[1] + p[2]


def p_repeat_list(p):
    '''repeat_list : param_list INT_NUMBER R'''
    last_value = p[1][-1]
    p[0] = p[1] + [last_value] * p[2]


def p_multiply_list(p):
    '''multiply_list : param_list INT_NUMBER M'''
    last_value = p[1][-1]
    p[0] = p[1] + [last_value * p[2]]


def p_interpolate_list(p):
    '''interpolate_list : param_list INT_NUMBER I float'''
    start = p[1][-1]
    finish = p[4]
    step = (finish - start) / p[2]
    p[0] = p[1] + [start + i * step for i in range(p[2] + 2)]


def p_skip_list(p):
    '''skip_list : INT_NUMBER J
                 | J'''
    repeat = p[2] if len(p) == 3 else 1
    p[0] = [None] * repeat


def p_float(p):
    '''float : '+' FLT_NUMBER
             | '-' FLT_NUMBER
             | FLT_NUMBER
             | integer'''
    if p[1] == '-':
        p[0] = -p[2]
    elif p[1] == '+':
        p[0] = p[2]
    else:
        p[0] = p[1]



#parser = yacc.yacc()

with open('..\\experiments\\test2.i') as f:
    text = f.read()
    lexer.input(text.upper())
    while True:
        tok = lexer.token()
        print(tok)
    result = parser.parse(text.upper())
    print(result)
