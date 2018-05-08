"""Functions for printing."""
import warnings


MCNP_FORMATS = {
    'importance': '{0:.3f}'
}


def print_card(card, offset=8, maxcol=80, sep='\n'):
    """Produce string in MCNP card format.

    Parameters
    ----------
    card : list[str]
        List of words to be printed.
    offset : int
        The number of spaces to make continuation of line. Minimum 5.
    maxcol : int
        The maximum length of card line. Maximum 80.
    sep : str
        Separator symbol. This symbol marks positions where newline character
        should be inserted even if maxcol position not reached.

    Returns
    -------
    text : str
        Text string that describes the card.
    """
    if offset < 5:
        offset = 5
        warnings.warn("offset must not be less than 5. offset is set to be 5.")
    if maxcol > 80:
        maxcol = 80
        warnings.warn("maxcol must not be greater than 80. It is set to be 80.")

    length = 0  # current length.
    words = []  # a list of individual words.
    line_sep = '\n' + ' ' * offset   # separator between lines.
    i = 0
    while i < len(card):
        if length + len(card[i]) > maxcol or card[i] == sep:
            words.append(line_sep)
            length = offset
            while card[i] == sep or card[i].isspace():
                i += 1
                if i == len(card):
                    words.pop()
                    return ''.join(words)
        words.append(card[i])
        length += len(card[i])
        i += 1
    return ''.join(words)


def print_option(option, value):
    name = option[:3]
    par = option[3:]
    if name == 'IMP' and (par == 'N' or par == 'P' or par == 'E'):
        return ['IMP:{0}={1}'.format(par, MCNP_FORMATS['importance'].format(value))]
    elif option == 'U':
        return ['U={0}'.format(value.name)]
    elif option == 'FILL':
        universe = value
        return ['FILL={0}'.format(universe.name)]
    else:
        raise ValueError("Incorrect option name: {0}".format(option))


CELL_OPTION_GROUPS = (
    ('IMPN', 'IMPP', 'IMPE'),   # Importance options
    ('TRCL',),  # Transformation options
    ('U', 'FILL')  # Universe and fill options
)
