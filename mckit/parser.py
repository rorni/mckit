# -*- coding: utf-8 -*-

import re

import ply.lex as lex
import ply.yacc as yacc


literals = ['+', '-', ':', '*', '(', ')', '#', ',', '.']

reserved = {
    'IMP', 'VOL', 'PWT', 'EXT', 'FCL', 'WWN', 'DXC', 'NONU', 'PD', 'TMP', 'U',
    'TRCL', 'LAT', 'FILL', 'LIKE', 'BUT', 'MAT', 'RHO',
    'MODE', 'AREA', 'TR', 'ESPLT', 'TSPLT', 'VECT', 'WWE', 'WWP', 'WWG', 'WWGE',
    'MESH', 'BBREM',
    'SDEF', 'SI', 'SP', 'SB', 'DS', 'SC', 'SSW', 'SSR', 'KCODE', 'KSRC', 'ERG',
    'TME', 'UUU', 'VVV', 'WWW', 'XXX', 'YYY', 'ZZZ', 'IPT', 'WGT', 'ICL', 'JSU',
    'NRM', 'POS', 'RAD', 'X', 'Y', 'Z', 'CCC', 'ARA', 'EFF', 'PAR',
    'F', 'FC', 'E', 'T', 'C', 'FQ', 'FM', 'DE', 'DF', 'EM', 'TM', 'CM', 'CF',
    'SF', 'FS', 'SD', 'FU', 'TF', 'DD', 'DXT', 'FT', 'FMESH', 'GEOM', 'ORIGIN',
    'AXS', 'VEC', 'IMESH', 'IINTS', 'JMESH', 'JINTS', 'KMESH', 'KINTS', 'EMESH',
    'EINTS', 'FACTOR', 'OUT', 'M', 'MPN', 'DRXS', 'TOTNU', 'AWTAB', 'XS',
    'VOID', 'PIKMT', 'MGOPT', 'GAS', 'ESTEP', 'NLIB', 'PLIB', 'PNLIB', 'ELIB',
    'COND', 'N', 'P', 'E',
    'P', 'PX', 'PY', 'PZ', 'S', 'SO', 'SX', 'SY', 'SZ', 'CX', 'CY', 'CZ', 'KX',
    'KY', 'KZ', 'TX', 'TY', 'TZ', 'C_X', 'C_Y', 'C_Z', 'K_X', 'K_Y', 'K_Z',
    'SQ', 'GQ',
    'PHYS', 'THTME', 'MT', 'CUT', 'ELPT', 'NOTRN', 'NPS', 'CTME', 'IDUM',
    'RDUM', 'PRDMP', 'LOST', 'DBCN', 'FILES', 'PRINT', 'TALNP', 'MPLOT',
    'PTRAC', 'PERT',
    'R', 'I', 'ILOG', 'J', 'NO', 'MESSAGE'
}

# List of token names
tokens = [
    'BLANK_LINE_DELIMITER',
    'LINE_COMMENT',
    'CARD_COMMENT',
    'CONTINUE',
    'NEWLINE',
    'SKIP_NEWLINE',
    'INT_NUMBER',
    'FLT_NUMBER',
    'INT_ZERO',
    'KEYWORD',
    'TITLE'
] + list(reserved)


states = (
    ('continue', 'inclusive'),
    ('title', 'exclusive')
)


def t_title_TITLE(t):
    r'^.+'
    t.lexer.lineno += 1
    t.lexer.begin('INITIAL')
    return t


def t_BLANK_LINE_DELIMITER(t):
    r'^[ ]*\n'
    t.lexer.lineno += 1
    return t


def t_LINE_COMMENT(t):
    r'^[ ]{0,4}C( .*)?'
    pass  # return t


def t_CARD_COMMENT(t):
    r'\$.*'
    pass  # return t


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


def t_CONTINUE(t):
    r'&'
    t.lexer.begin('continue')


def t_continue_SKIP_NEWLINE(t):
    r'\n(?=[ ]{0,4}C( .*)?)'
    t.lexer.lineno += 1
    # return t


def t_continue_NEWLINE(t):
    r'\n(?![ ]{0,4}C( .*)?)'
    t.lexer.lineno += 1
    t.lexer.begin('INITIAL')
    # t.value += 'continue'
    # return t


def t_SKIP_NEWLINE(t):
    r'\n(?=[ ]{5,}[^\s]|[ ]{0,4}C( .*)?)'
    t.lexer.lineno += 1
    # return t


def t_NEWLINE(t):
    r'\n'
    t.lexer.lineno += 1
    return t


def t_FLT_NUMBER(t):
    r'\d*\.\d+(E[-+]?\d+)?|\d+\.?E[-+]?\d+'
    t.value = float(t.value)
    return t

def t_INT_ZERO(t):
    r'0'
    t.value = 0
    return t

def t_INT_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t


def t_KEYWORD(t):
    r'[A-Z]+(/[A-Z]+)?'
    if '/' in t.value:
        t.value = t.value.replace('/', '_')
    if t.value in reserved:
        t.type = t.value
    else:
        raise ValueError('Unknown word' + t.value)
    return t


def t_ignore_SKIP(t):
    r'[ ]+|='
    pass


lexer = lex.lex(reflags=re.MULTILINE + re.IGNORECASE + re.VERBOSE)
lexer.begin('title')


#def p_model(p):
#    '''model : message_block BLANK_LINE_DELIMITER model_body
#             | model_body
#    '''
#    pass


def p_model_body(p):
    '''model_body : TITLE NEWLINE cell_cards BLANK_LINE_DELIMITER \
                    surface_cards BLANK_LINE_DELIMITER \
                    data_cards BLANK_LINE_DELIMITER'''
    p[0] = p[1], p[3], p[5], p[7]


def p_cell_cards(p):
    '''cell_cards : cell_cards cell_card
                  | cell_card
    '''
    name = p[1][0]
    params = p[1][1:]
    if len(p) == 3:
        p[0] = {name: params}
    elif len(p) == 4:
        p[3][name] = params
        p[0] = p[3]


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
    p[0] = 'IMP', p[3], p[4]


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
    p[0] = p[1] + p[3]
    p[0].append('U')


def p_term(p):
    '''term : term factor
            | factor'''
    p[0] = p[1] + p[2]
    p[0].append('I')


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
        name, *params = p[2]
        p[2][name] = params
        p[0] = p[2]


def p_data_cards(p):
    '''data_cards : data_cards data_card
                  | data_card
    '''
    pass


def p_data_card(p):
    '''data_card : mode_card
                 | volume_card'''
    p[0] = p[1]


def p_mode_card(p):
    '''mode_card : MODE particle_list'''
    p[0] = 'MODE', p[2]


def p_particle_list(p):
    '''particle_list : particle_list particle
                     | particle'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]


def p_volume_card(p):
    '''volume_card : VOL integer_list
                   | VOL NO integer_list'''
    if len(p) == 3:
        p[0] = 'VOL', p[2]
    else:
        p[0] = 'VOL', p[3]


def p_integer_list(p):
    '''integer_list : integer_list INT_NUMBER
                    | INT_NUMBER'''
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
                    | INT_NUMBER surface_description NEWLINE'''
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
        if isinstance(p[1], float):
            p[0] = [p[1]]
        else:
            p[0] = p[1]
    else:
        if isinstance(p[2], float):
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



parser = yacc.yacc()

with open('c:\\Users\\Roma\\projects\\UPP02\\upp02_m9.i') as f:
    text = f.read()
#    lexer.input(text)
#    while True:
#        tok = lexer.token()
#        print(tok)
    result = parser.parse(text.upper())
    print(result)
