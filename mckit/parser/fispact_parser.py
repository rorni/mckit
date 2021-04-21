import re

import ply.lex as lex
import ply.yacc as yacc

from mckit.constants import TIME_UNITS
from mckit.material import Element

TIME_ALIAS = {
    "s": TIME_UNITS["SECS"],
    "m": TIME_UNITS["MINS"],
    "h": TIME_UNITS["HOURS"],
    "d": TIME_UNITS["DAYS"],
    "y": TIME_UNITS["YEARS"],
    "ky": TIME_UNITS["YEARS"] * 1000,
}

literals = ["+", "-", "(", ")"]


# List of token names
tokens = ["newline", "int_number", "flt_number", "keyword", "TIME", "INTERVAL"]


NEWLINE = r"\n"
EXPONENT = r"(E[-+]?\d+)"
INT_NUMBER = r"(\d+)"
FLT_NUMBER = (
    INT_NUMBER
    + r"?"
    + r"\."
    + INT_NUMBER
    + EXPONENT
    + r"?|"
    + INT_NUMBER
    + r"\."
    + r"?"
    + EXPONENT
    + r"|"
    + INT_NUMBER
    + r"\."
)
KEYWORD = r"[A-Z]+(/[A-Z]+)?"
SKIP = r"[. ]"

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
    if value == "TIME" or value == "INTERVAL":
        t.type = value
        t.value = value
    return t


def t_error(t):
    column = t.lexer.lexpos - t.lexer.last_pos + 1
    msg = r"Illegal character '{0}' at line {1} column {2}".format(
        t.value[0], t.lexer.lineno, column
    )
    raise ValueError(msg, t.value[0], t.lexer.lineno, column)


fispact_lexer = lex.lex(reflags=re.MULTILINE + re.IGNORECASE + re.VERBOSE)


def p_data(p):
    """data : data timeframe
    | timeframe
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[1].append(p[2])
        p[0] = p[1]


def p_timeframe(p):
    """timeframe : timeheader isotope_data
    | timeheader gamma_data
    | timeheader
    """
    index, interval, time = p[1]
    frame = {"index": index, "duration": interval, "time": time}
    if len(p) == 3:
        frame.update(p[2])
    p[0] = frame


def p_timeheader(p):
    """timeheader : TIME flt_number keyword INTERVAL int_number INTERVAL TIME flt_number keyword newline"""
    index = p[5]
    time = p[2] * TIME_ALIAS[p[3]]
    interval = p[8] * TIME_UNITS[p[9]]
    p[0] = index, interval, time


def p_isotope_data(p):
    """isotope_data : isotope_data isotope_row newline
    | isotope_row newline
    """
    n = len(p)
    elem, data1, data2 = p[n - 2]
    if n == 3:
        p[0] = {"data1": {elem: data1}, "data2": {elem: data2}}
    else:
        p[1]["data1"][elem] = data1
        p[1]["data2"][elem] = data2
        p[0] = p[1]


def p_isotope_row(p):
    """isotope_row : isotope flt_number flt_number
    | isotope int_number flt_number flt_number
    """
    n = len(p)
    p[0] = p[1], p[n - 2], p[n - 1]


def p_isotope(p):
    """isotope : keyword int_number
    | keyword int_number keyword
    """
    name = p[1] + str(p[2])
    if len(p) == 4:
        isomer = ord(p[3]) - ord("m") + 1
    else:
        isomer = 0
    p[0] = Element(name, isomer=isomer)


def p_gamma_data(p):
    """gamma_data : fiss ab ab tot_gamma spectrum"""
    p[5]["fissions"] = p[1]
    p[5]["a-energy"] = p[2]
    p[5]["b-energy"] = p[3]
    p[5]["g-energy"] = p[4]
    p[0] = p[5]


def p_fiss(p):
    """fiss : keyword keyword keyword flt_number newline"""
    p[0] = p[4]


def p_ab(p):
    """ab : keyword '-' keyword keyword flt_number newline"""
    p[0] = p[5]


def p_tot_gamma(p):
    """tot_gamma : keyword keyword keyword flt_number newline"""
    p[0] = p[4]


def p_spectrum(p):
    """spectrum : spectrum bin newline
    | bin newline
    """
    n = len(p)
    low, high, data1, data2 = p[n - 2]
    if n == 3:
        p[0] = {"ebins": [low, high], "data1": [data1], "data2": [data2]}
    else:
        p[1]["ebins"].append(high)
        p[1]["data1"].append(data1)
        p[1]["data2"].append(data2)
        p[0] = p[1]


def p_bin(p):
    """bin : '(' flt_number '-' flt_number keyword ')' flt_number flt_number"""
    p[0] = p[2], p[4], p[7], p[8]


def p_error(p):
    if p:
        column = p.lexer.lexpos - p.lexer.last_pos + 1
        print(
            "Syntax error at token {0} {3}, line {1}, column {2}".format(
                p.type, p.lexer.lineno, column, p.value
            )
        )
    else:
        print("Syntax error at EOF")


fispact_parser = yacc.yacc(tabmodule="fispact_tab", debug=True)


def read_fispact_tab(filename):
    """Reads FISPACT output tab file.

    This function is common for all tab files (1-4). Output data format depends
    on file extension: tab1 - number of atoms of each nuclide, tab2 - activity
    of each nuclide, tab3 - ingestion and inhalation doses,
    tab4 - gamma-radiation spectrum.

    The result is a list of time frames. Every time frame is a dictionary, that
    contains the following data:
        * index - the number of frame. Numbers start from 1;
        * time - total time passed from the start of irradiation or relaxation
                 in seconds;
        * duration - time passed from the previous frame also in seconds;
        * atoms - dictionary[Element -> float] - the number of atoms for each
                 element. Only for tab1 file;
        * activity - dictionary[Element -> float] - the activity of each
                 element in Bq. Only for tab2 file;
        * ingestion - dictionary[Element -> float] - ingestion dose for every
                 isotope in Sv/h. Only for tab3 file;
        * inhalation - dictionary[Element -> float] - inhalation dose for every
                 isotope in Sv/h. Only for tab3 file;
        * fissions - the number of spontaneous fissions. Only for tab4 file;
        * a-energy - energy of alpha radiation in MeV/sec. Only for tab4 file.
        * b-energy - energy of beta radiation in MeV/sec. Only for tab4 file.
        * g-energy - energy of gamma radiation in MeV/sec. Only for tab4 file.
        * ebins - energy bins for gamma-radiation in MeV. Only for tab4 file.
        * flux - gamma-radiation group flux in photons per cc per sec. Only for
                 tab4 file.

    Parameters
    ----------
    filename : str
        Name of FISPACT output tab file.

    Returns
    -------
    time_frames : list
        List of time frames.
    """
    ext = filename.rpartition(".")[2].lower()
    with open(filename) as f:
        text = f.read()
    fispact_lexer.lineno = 1
    fispact_lexer.linepos = 1
    fispact_lexer.last_pos = 1
    time_frames = fispact_parser.parse(text, lexer=fispact_lexer)
    time = 0
    for tf in time_frames:
        time += tf["duration"]
        tf["time"] = time
        data1 = tf.pop("data1", None)
        data2 = tf.pop("data2", None)
        if data1 is None and data2 is None:
            continue
        if ext == "tab1":
            tf["atoms"] = data1
        elif ext == "tab2":
            tf["activity"] = data1
        elif ext == "tab3":
            tf["ingestion"] = data1
            tf["inhalation"] = data2
        elif ext == "tab4":
            tf["flux"] = data2
    return time_frames
