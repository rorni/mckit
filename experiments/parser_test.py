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
    'NUMBER'
)

literals = ['+', '-', ':', '(', ')']

states = (
    ('continue', 'inclusive'),
)


def t_BLANK_LINE_DELIMITER(t):
    r'^[ ]*\n'
    t.lexer.lineno += 1
    return t

def t_LINE_COMMENT(t):
    r'^[ ]{0,4}C( .*)?'
    pass #return t

def t_CARD_COMMENT(t):
    r'\$.*'
    pass #return t

def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

def t_CONTINUE(t):
    r'&'
    t.lexer.begin('continue')

def t_continue_SKIP_NEWLINE(t):
    r'\n(?=[ ]{0,4}C( .*)?)'
    t.lexer.lineno += 1
    #return t

def t_continue_NEWLINE(t):
    r'\n(?![ ]{0,4}C( .*)?)'
    t.lexer.lineno += 1
    t.lexer.begin('INITIAL')
    #t.value += 'continue'
    #return t

def t_SKIP_NEWLINE(t):
    r'\n(?=[ ]{5,}[^\s]|[ ]{0,4}C( .*)?)'
    t.lexer.lineno += 1
    #return t

def t_NEWLINE(t):
    r'\n'
    t.lexer.lineno += 1
    return t


def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_ignore_SKIP(t):
    r'[ ]+'
    pass


lexer = lex.lex(reflags=re.MULTILINE+re.IGNORECASE+re.VERBOSE)

with open('test1.i') as f:
    text = f.read()

lexer.input(text)
while True:
    tok = lexer.token()
    print(tok)