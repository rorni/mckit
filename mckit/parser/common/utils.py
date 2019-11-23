import re

RE_C_COMMENT = re.compile(r'\n\s{0,5}c\s[^\n]*', re.IGNORECASE | re.MULTILINE)
TRAIL_COMMENT = r'\n\s+\$[^\n]*\n?'
EOL_COMMENT = r'\$[^\n]*\n?'
FLOAT = r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?'
INTEGER = r'\d+'


def ensure_lower(text: str):
    if not text.islower():
        text = text.lower()
    return text


def ensure_upper(text: str):
    if not text.isupper():
        text = text.lower()
    return text


def drop_c_comments(text):
    if RE_C_COMMENT.search(text) is not None:
        text = RE_C_COMMENT.sub('', text)
    return text


class ParseError(ValueError):
    """Parsing exception"""


# noinspection PyPep8Naming,PyUnboundLocalVariable,PyUnresolvedReferences,SpellCheckingInspection

