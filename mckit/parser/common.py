import re

RE_C_COMMENT = re.compile(r'\n\s{,5}c\s[^\n]*', re.IGNORECASE | re.MULTILINE)

INTEGER = r'\d+'
FLOAT = r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?'
TRAIL_COMMENT = r'\n\s+\$[^\n]*\n?'
EOL_COMMENT = r'\$[^\n]*\n?'


def drop_c_comments(text):
    if RE_C_COMMENT.search(text) is not None:
        text = RE_C_COMMENT.sub('', text)
    return text
