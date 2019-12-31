import re

C_COMMENT = r'\n\s{0,5}[cC]\s[^\n]*'
RE_C_COMMENT = re.compile(C_COMMENT, re.MULTILINE)
EOL_COMMENT = r'\$.*[^\n]*'
RE_EOL_COMMENT = re.compile(EOL_COMMENT, re.MULTILINE)
LINE = r'(?P<text>[^$]+)?(?:\s+\$\s*(?P<comment>.*))?'
RE_LINE = re.compile(LINE)
FLOAT = r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?'
INTEGER = r'\d+'


def ensure_lower(text: str):
    if not text.islower():
        text = text.lower()
    return text


def ensure_upper(text: str):
    if not text.isupper():
        text = text.upper()
    return text


def drop_c_comments(text):
    if RE_C_COMMENT.search(text) is not None:
        text = RE_C_COMMENT.sub('', text)
    return text


def drop_eol_comments(text):
    if RE_EOL_COMMENT.search(text) is not None:
        text = RE_EOL_COMMENT.sub('', text)
    return text


def drop_comments(text):
    return drop_eol_comments(drop_c_comments(text))


def extract_comments(text):
    lines = text.split('\n')
    cleaned_text = []
    comments = []
    trailing_comment = []

    def add_trailing_to_previous_item():
        nonlocal comments, trailing_comment
        if trailing_comment:
            assert 0 < len(comments), "The comments should not be empty on this call: at least card name is read"
            comments[-1][1].extend(trailing_comment)
            trailing_comment = []

    for i, line in enumerate(lines):
        match = RE_LINE.fullmatch(line)
        assert match is not None
        groupdict = match.groupdict()
        t = groupdict["text"]
        c = groupdict["comment"]
        if t:
            cleaned_text.append(t)
            add_trailing_to_previous_item()
            if c is not None:
                comments.append((i + 1, [c]))  # lexer counts lines from 1
        else:
            assert c is not None, "If there's no text, then at least comment should present"
            trailing_comment.append(c)

    assert cleaned_text, "There should be some  text in a card"

    if comments:
        comments = dict((k, tuple(v)) for k, v in comments)
    else:
        comments = None

    if not trailing_comment:
        trailing_comment = None

    return '\n'.join(cleaned_text), comments, trailing_comment


class ParseError(ValueError):
    """Parsing exception"""


# noinspection PyPep8Naming,PyUnboundLocalVariable,PyUnresolvedReferences,SpellCheckingInspection

