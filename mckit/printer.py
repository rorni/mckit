"""Functions for printing."""
import warnings


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
    while i < len(card) - 1:
        if length + len(card[i]) > maxcol or card[i] == sep:
            words.append(line_sep)
            length = offset
            while card[i] == sep or card[i].isspace():
                i += 1
                if i == len(card):
                    words.pop()
                    break
        words.append(card[i])
        length += len(card[i])
        i += 1
    return ''.join(words)
