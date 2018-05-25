import re

import numpy as np

import ply.lex as lex
import ply.yacc as yacc

literals = ['+', '-', ':', '/']

KEYWORDS = [
    'MCNP', 'VERSION', 'LD', 'PROBID', 'NEUTRON', 'PHOTON', 'ELECTRON', 'RESULT', 'RESULTS', 'ERROR', 'ERRORS',
    'CYLINDER', 'ORIGIN', 'AXIS', 'TALLY',
    'X', 'Y', 'Z', 'R', 'THETA', 'ENERGY', 'TH', 'TOTAL'
]

BIN_REC_ORDER = {'ENERGY': 0, 'X': 1, 'Y': 2, 'Z': 3}
BIN_CYL_ORDER = {'ENERGY': 0, 'R': 1, 'Z': 2, 'THETA': 3}

# List of token names
tokens = [
    'newline',
    'int_number',
    'flt_number',
    'minus',
    'title',
    'stamp',
    'MCNP', 'VERSION', 'LD', 'PROBID', 'NEUTRON', 'PHOTON', 'ELECTRON', 'RESULT',  'ERROR',
    'CYLINDER', 'ORIGIN', 'AXIS', 'TALLY',
    'X', 'Y', 'Z', 'R', 'THETA', 'ENERGY', 'TOTAL'
]

precedence = (
    ('left', 'newline'),
)

states = (
    ('title', 'exclusive'),
    ('norm', 'exclusive'),
    ('tally', 'exclusive')
)

BLANK_LINE = r'^\n'
STAMP = r'[\d/:]+'
NEWLINE = r'\n'
FRACTION = r'\.'
EXPONENT = r'(E[-+]?\d+)'
INT_NUMBER = r'(\d+)'
MINUS = r'-(?=\d)'
FLT_NUMBER = r'(' + \
             INT_NUMBER + r'?' + FRACTION + INT_NUMBER + EXPONENT + r'?|' +\
             INT_NUMBER + FRACTION + r'?' + EXPONENT + r'|' + \
             INT_NUMBER + FRACTION + r')(?=[ \n-+])'
KEYWORD = r'[A-Z]+(/[A-Z]+)?'

SKIP = r'[=,.() ]'

t_ANY_ignore = SKIP


def t_title_title(t):
    r""".+"""
    return t


@lex.TOKEN(MINUS)
def t_norm_tally_minus(t):
    return t


@lex.TOKEN(BLANK_LINE)
def t_ANY_blank_line(t):
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos


@lex.TOKEN(NEWLINE)
def t_newline(t):
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos
    t.lexer.begin('title')
    return t


@lex.TOKEN(NEWLINE)
def t_title_newline(t):
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos
    t.lexer.begin('norm')
    return t


@lex.TOKEN(NEWLINE)
def t_norm_newline(t):
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos
    t.lexer.begin('tally')
    return t


@lex.TOKEN(NEWLINE)
def t_tally_newline(t):
    t.lexer.lineno += 1
    t.lexer.last_pos = t.lexer.lexpos
    return t


@lex.TOKEN(STAMP)
def t_stamp(t):
    return t


@lex.TOKEN(KEYWORD)
def t_keyword(t):
    value = t.value.upper()
    if value == 'MCNP':
        t.lexer.section_index = 0
        meshtal_lexer.lineno = 1
        t.lineno = 1
        t.lexer.last_pos = 1
    if value in KEYWORDS:
        t.type = value
        t.value = value
        return t


@lex.TOKEN(KEYWORD)
def t_norm_tally_keyword(t):
    value = t.value.upper()
    if value in KEYWORDS:
        if value == 'RESULTS':
            value = 'RESULT'
        elif value == 'ERRORS':
            value = 'ERROR'
        elif value == 'TH':
            value = 'THETA'
        t.type = value
        t.value = value
        return t


def t_ANY_error(t):
    column = t.lexer.lexpos - t.lexer.last_pos + 1
    msg = r"Illegal character '{0}' at line {1} column {2}".format(
        t.value[0], t.lexer.lineno, column)
    raise ValueError(msg, t.value[0], t.lexer.lineno, column)


@lex.TOKEN(FLT_NUMBER)
def t_norm_tally_flt_number(t):
    t.value = float(t.value)
    return t


@lex.TOKEN(INT_NUMBER)
def t_norm_tally_int_number(t):
    t.value = int(t.value)
    return t


meshtal_lexer = lex.lex(reflags=re.MULTILINE + re.IGNORECASE + re.VERBOSE)


def p_meshtal(p):
    """meshtal : header newline title newline float newline tallies
               | header newline title newline float newline tallies newline
    """
    p[0] = {'date': p[1]['PROBID'], 'title': p[3], 'histories': p[5], 'tallies': p[7]}


def p_header(p):
    """header : MCNP VERSION stamp LD stamp PROBID stamp stamp"""
    p[0] = {'PROBID': p[7] + p[8], 'VERSION': p[3]}


def p_tallies(p):
    """tallies : tallies newline tally
               | tally
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[1].append(p[3])
        p[0] = p[1]


def p_tally(p):
    """tally :  TALLY integer newline particle TALLY newline boundaries newline data"""
    tally = {'name': p[2], 'particle': p[4]}
    boundaries = p[7]
    tally['geom'] = 'XYZ'
    if 'ORIGIN' in boundaries.keys():
        tally['origin'] = boundaries.pop('ORIGIN')
        tally['geom'] = 'CYL'
    if 'AXIS' in boundaries.keys():
        tally['axis'] = boundaries.pop('AXIS')
    tally['bins'] = boundaries.copy()
    data = p[9]
    od = BIN_CYL_ORDER if 'origin' in tally.keys() else BIN_REC_ORDER
    if 'result' in data.keys():
        src_perm = [od[let] for let in data['order']]
        tally['result'] = np.moveaxis(np.array(data['result']), src_perm, (0, 1, 2, 3))
        tally['error'] = np.moveaxis(np.array(data['result']), src_perm, (0, 1, 2, 3))
    else:
        header = data['header']
        data = np.array(data['data'])
        boundaries['ENERGY'] = 0.5 * (boundaries['ENERGY'][2:] + boundaries['ENERGY'][1:-1])
        shape = [0, 0, 0, 0]
        for k, v in od.items():
            shape[v] = len(boundaries[k]) - 1
        result = np.empty(shape)
        error = np.empty(shape)
        indices = []
        for k in od.keys():
            if k in header:
                indices[od[k]] = np.searchsorted(boundaries[k], data[:, header.index(k)])
            else:
                indices[od[k]] = np.zeros(data.shape[0])
        res_ind = header.index('RESULT')
        err_ind = header.index('ERROR')
        indices = np.array(indices)
        for i in range(indices.shape[1]):
            result[indices[:, i]] = data[i, res_ind]
            error[indices[:, i]] = data[i, err_ind]
        tally['result'] = result
        tally['error'] = error
    p[0] = tally


def p_particle(p):
    """particle : NEUTRON
                | PHOTON
                | ELECTRON
    """
    p[0] = p[1]


def p_boundaries(p):
    """boundaries : TALLY ':' newline CYLINDER ORIGIN vector AXIS vector newline bins
                  | TALLY ':' newline bins
    """
    boundaries = p[len(p) - 1]
    if len(p) == 11:
        boundaries['ORIGIN'] = p[6]
        boundaries['AXIS'] = p[8]
    p[0] = boundaries


def p_bins(p):
    """bins : direction newline direction newline direction newline energies"""
    bins = {}
    for name, data in [p[1], p[3], p[5], p[7]]:
        bins[name] = data
    p[0] = bins


def p_dir_spec(p):
    """dir_spec : X
                | Y
                | Z
                | R
                | THETA
    """
    p[0] = p[1]


def p_direction(p):
    """direction : dir_spec ':' vector
    """
    p[0] = p[1], p[3]


def p_energies(p):
    """energies : ENERGY ':' vector"""
    p[0] = p[1], p[3]


def p_float(p):
    """float : '+' flt_number
             | minus flt_number
             | flt_number
             | integer
    """
    if p[1] == '-':
        p[0] = -p[2]
    elif p[1] == '+':
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_integer(p):
    """integer : '+' int_number
               | minus int_number
               | int_number
    """
    if p[1] == '-':
        p[0] = -p[2]
    elif p[1] == '+':
        p[0] = p[2]
    else:
        p[0] = p[1]


def p_vector(p):
    """vector : vector float
              | float
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[1].append(p[2])
        p[0] = p[1]


def p_matrix(p):
    """matrix : matrix newline vector
              | vector
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[1].append(p[3])
        p[0] = p[1]


def p_total_matrix(p):
    """total_matrix : total_matrix newline TOTAL vector
                    | TOTAL vector"""
    if len(p) == 2:
        p[0] = [p[2]]
    else:
        p[1].append(p[4])
        p[0] = p[1]


def p_data(p):
    """data : matrix_data
            | column_data
    """
    p[0] = p[1]


def p_column_data(p):
    """column_data : column_header newline matrix newline total_matrix
                   | column_header newline matrix
    """
    p[0] = {'header': p[1], 'data': p[3]}
    if len(p) == 6:
        p[0]['total'] = p[5]


def p_column_header(p):
    """column_header : ENERGY dir_spec dir_spec dir_spec RESULT ERROR
                     | dir_spec dir_spec dir_spec RESULT ERROR
    """
    p[0] = p[1:]


def p_matrix_data(p):
    """matrix_data : energy_bins newline total_energy_bin
                   | energy_bins
    """
    order, result, error = p[1]
    p[0] = {'result': result, 'error': error, 'order': order}


def p_energy_bins(p):
    """energy_bins : energy_bins newline energy_bin
                   | energy_bin"""
    if len(p) == 2:
        order, result, error = p[1]
        p[0] = order, [result], [error]
    else:
        order, result, error = p[3]
        p[1][1].append(result)
        p[1][2].append(error)
        p[0] = order, p[1][1], p[1][2]


def p_energy_bin(p):
    """energy_bin : ENERGY ':' float '-' float newline spatial_bins"""
    order = [p[1]] + p[7][0]
    p[0] = order, p[7][1], p[7][2]


def p_total_energy_bin(p):
    """total_energy_bin : TOTAL ENERGY newline spatial_bins"""
    order = [p[1]] + p[4][0]
    p[0] = order, p[4][1], p[4][2]


def p_spatial_bins(p):
    """spatial_bins : spatial_bins newline spatial_bin
                    | spatial_bin
    """
    if len(p) == 2:
        order, result, error = p[1]
        p[0] = order, [result], [error]
    else:
        order, result, error = p[3]
        p[1][1].append(result)
        p[1][2].append(error)
        p[0] = order, p[1][1], p[1][2]


def p_spatial_bin(p):
    """spatial_bin : dir_spec ':' float '-' float newline TALLY RESULT ':' dir_spec dir_spec newline matrix newline ERROR newline matrix"""
    order = [p[1], p[11], p[10]]
    results = [line[1:] for line in p[13][1:]]
    errors = [line[1:] for line in p[17][1:]]
    p[0] = order, results, errors


def p_error(p):
    if p:
        column = p.lexer.lexpos - p.lexer.last_pos + 1
        print("Syntax error at token {0} {3}, line {1}, column {2}".format(p.type, p.lexer.lineno, column, p.value))
    else:
        print("Syntax error at EOF")


meshtal_parser = yacc.yacc(tabmodule="meshtal_tab", debug=True)
