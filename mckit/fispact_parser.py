import re

import numpy as np

import ply.lex as lex
import ply.yacc as yacc

from .activation import TIME_UNITS
from .material import Element


TIME_ALIAS = {
    's': TIME_UNITS['SECS'],
    'm': TIME_UNITS['MINS'],
    'h': TIME_UNITS['HOURS'],
    'd': TIME_UNITS['DAYS'],
    'y': TIME_UNITS['YEARS'],
    'ky': TIME_UNITS['YEARS'] * 1000
}

literals = ['+', '-', ':', '(', ')']


# List of token names
tokens = [
    'newline',
    'int_number',
    'flt_number',
    'keyword',
    'TIME',
    'INTERVAL'
]


NEWLINE = r'\n'
EXPONENT = r'(E[-+]?\d+)'
INT_NUMBER = r'(\d+)'
FLT_NUMBER = r'(' + \
             INT_NUMBER + r'?' + r'\.' + INT_NUMBER + EXPONENT + r'?|' +\
             INT_NUMBER + r'\.' + r'?' + EXPONENT + r'|' + \
             INT_NUMBER + r'\.' + r')(?=[ \n-+])'
KEYWORD = r'[A-Z]+(/[A-Z]+)?'
SKIP = r'[. ]'

t_ignore = SKIP


@lex.TOKEN(NEWLINE)
def t_newline(t):
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos
    return t


@lex.TOKEN(FLT_NUMBER)
def t_flt_number(t):
    t.value = float(t.value)
    return t


@lex.TOKEN(INT_NUMBER)
def t_int_number(t):
    t.value = int(t.value)
    return t


@lex.TOKEN(KEYWORD)
def t_keyword(t):
    value = t.value.upper()
    if value == 'TIME' or value == 'INTERVAL':
        t.type = value
        t.value = value
    return t


def t_error(t):
    column = t.lexer.lexpos - t.lexer.last_pos + 1
    msg = r"Illegal character '{0}' at line {1} column {2}".format(
        t.value[0], t.lexer.lineno, column)
    raise ValueError(msg, t.value[0], t.lexer.lineno, column)


fispact_lexer = lex.lex(reflags=re.MULTILINE + re.IGNORECASE + re.VERBOSE)


def p_data(p):
    """data : data timeframe
            | timeframe
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[1].append(p[2])
        p[0] = p[1]


def p_timeframe(p):
    """timeframe : timeheader isotope_data
                 | timeheader gamma_data
                 | timeheader
    """
    index, interval, time = p[1]
    frame = {'index': index, 'duration': interval, 'final_time': time}
    if len(p) == 3:
        frame.update(p[2])
    p[0] = frame


def p_timeheader(p):
    """timeheader : TIME flt_number keyword INTERVAL int_number INTERVAL TIME flt_number keyword newline
    """
    index = p[5]
    time = p[2] * TIME_ALIAS[p[3]]
    interval = p[8] * TIME_UNITS[p[9]]
    p[0] = index, interval, time


def p_isotope_data(p):
    """isotope_data : isotope_data isotope_row newline
                    | isotope_row newline
    """
    n = len(p)
    elem, data1, data2 = p[n-2]
    if n == 3:
        p[0] = {'data1': {elem: data1}, 'data2': {elem: data2}}
    else:
        p[1]['data1'][elem] = data1
        p[1]['data2'][elem] = data2
        p[0] = p[1]


def p_isotope_row(p):
    """isotope_row : isotope flt_number flt_number
                   | isotope int_number flt_number flt_number
    """
    n = len(p)
    p[0] = p[1], p[n-2], p[n-1]


def p_isotope(p):
    """isotope : keyword int_number
               | keyword int_number keyword
    """
    name = p[1] + str(p[2])
    # TODO: add isomer interpretation
    p[0] = Element(name)


def p_gamma_data(p):
    """gamma_data : fiss ab ab tot_gamma spectrum"""
    p[5]['fissions'] = p[1]
    p[5]['alpha'] = p[2]
    p[5]['beta'] = p[3]
    p[5]['gamma'] = p[4]
    p[0] = p[5]


def p_fiss(p):
    """fiss : keyword '/' keyword keyword keyword flt_number newline"""
    p[0] = p[6]


def p_ab(p):
    """ab : keyword '-' keyword keyword '/' keyword flt_number newline"""
    p[0] = p[7]


def p_tot_gamma(p):
    """tot_gamma : keyword keyword keyword '/' keyword flt_number newline"""
    p[0] = p[6]


def p_spectrum(p):
    """spectrum : spectrum bin newline
                | bin newline
    """
    n = len(p)
    low, high, data1, data2 = p[n-2]
    if n == 3:
        p[0] = {'bins': [low, high], 'energy': [data1], 'spec': [data2]}
    else:
        p[1]['bins'].append(high)
        p[1]['energy'].append(data1)
        p[1]['spec'].append(data2)
        p[0] = p[1]


def p_bin(p):
    """bin : '(' flt_number '-' flt_number keyword ')' flt_number flt_number"""
    p[0] = p[2], p[4], p[7], p[8]


def p_error(p):
    if p:
        column = p.lexer.lexpos - p.lexer.last_pos + 1
        print("Syntax error at token {0} {3}, line {1}, column {2}".format(p.type, p.lexer.lineno, column, p.value))
    else:
        print("Syntax error at EOF")
