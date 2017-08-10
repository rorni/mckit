# -*- coding: utf-8 -*-

import re

import ply.lex as lex

# List of token names
tokens = (
    'BLANK_LINE_DELIMITER',
    'LINE_COMMENT',
    'CARD_COMMENT',
    'CONTINUE',
    'NEWLINE',
    'SKIP_NEWLINE',
    'INT_NUMBER',
    'FLT_NUMBER',
    'INT_ZERO',
    'SURFACE_SPEC',
    'KW_FLOAT',
    'KW_INT',
    'KW_COMPOUND',
    'KW_VECTOR3'
)

literals = ['+', '-', ':', '*', '(', ')', '#']

states = (
    ('continue', 'inclusive'),
)


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


def t_SURFACE_SPEC(t):
    r'[PTS][XYZ]|[CK]/?[XYZ]|[PS]|SO|[SG]Q'
    return t


def t_KW_INT(t):
    r'U|TRCL'
    return t


def t_KW_FLOAT(t):
    r'IMP:N|IMP:P|IMP:E|VOL'
    return t


def t_KW_VECTOR3(t):
    r'POS|VEC|AXS'
    return t


def t_KW_COMPOUND(t):
    r''


def t_ignore_SKIP(t):
    r'[ ]+|='
    pass


lexer = lex.lex(reflags=re.MULTILINE + re.IGNORECASE + re.VERBOSE)


def p_model(p):
    '''model : message_block BLANK_LINE_DELIMITER model_body
             | model_body
    '''
    pass


def p_model_body(p):
    '''model_body : cell_cards BLANK_LINE_DELIMITER
                    surface_cards BLANK_LINE_DELIMITER
                    data_cards BLANK_LINE_DELIMITER
    '''
    pass


def p_cell_cards(p):
    '''cell_cards : cell_card NEWLINE cell_cards | cell_card NEWLINE'''
    name = p[1][0]
    params = p[1][1:]
    if len(p) == 3:
        p[0] = {name: params}
    elif len(p) == 4:
        p[3][name] = params
        p[0] = p[3]


def p_cell_card(p):
    '''cell_card : INT_NUMBER cell_material expression options
                 | INT_NUMBER cell_material expression
                 | INT_NUMBER LIKE INT_NUMBER BUT options'''
    pass


def p_options(p):
    '''options : '''
    pass


def p_cell_material(p):
    '''cell_material : INT_ZERO | INT_NUMBER float'''
    if len(p) == 2:
        p[0] = None
    else:
        p[0] = p[1], p[2]


def p_expression(p):
    '''expression : term ':' expression'''
    p[0] = p[1] + p[3]
    p[0].append('U')


def p_term(p):
    '''term : factor term'''
    p[0] = p[1] + p[2]
    p[0].append('I')


def p_factor(p):
    '''factor : '#' '(' expression ')' | '(' expression ')'
              | '-' INT_NUMBER | '+' INT_NUMBER | '#' INT_NUMBER
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
    '''surface_cards : surface_card NEWLINE surface_cards | surface_card NEWLINE
    '''
    name = p[1][0]
    params = p[1][1:]
    if len(p) == 3:
        p[0] = {name: params}
    elif len(p) == 4:
        p[3][name] = params
        p[0] = p[3]


def p_data_cards(p):
    '''data_cards : data_card NEWLINE data_cards | data_card NEWLINE'''
    pass


def p_surface_card(p):
    '''surface_card : '*' INT_NUMBER surface_description
                    | '+' INT_NUMBER surface_description
                    | INT_NUMBER surface_description'''
    if len(p) == 3:
        p[0] = p[1], *p[2], None
    else:
        p[0] = p[2], *p[3], p[1]


def p_surface_description(p):
    '''surface_description : integer SURFACE_SPEC surface_params
                           | SURFACE_SPEC surface_params'''
    if len(p) == 4:
        p[0] = (p[2], p[3], p[1])
    else:
        p[0] = (p[1], p[2], None)


def p_integer(p):
    '''integer : '+' INT_NUMBER | '-' INT_NUMBER | '+' INT_ZERO | '-' INT_ZERO
               | INT_NUMBER | INT_ZERO
    '''
    if p[1] == '-':
        p[0] = -p[2]
    elif p[1] == '+':
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_surface_params(p):
    '''surface_params : float surface_params | float'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[2]


def p_float(p):
    '''float : '+' FLT_NUMBER | '-' FLT_NUMBER | FLT_NUMBER | integer'''
    if p[1] == '-':
        p[0] = -p[2]
    elif p[1] == '+':
        p[0] = p[2]
    else:
        p[0] = p[1]
