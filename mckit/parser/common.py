import re

RE_C_COMMENT = re.compile(r'\nc\s{1,5}[^\n]*', re.IGNORECASE | re.MULTILINE)


def drop_c_comments(text):
    if RE_C_COMMENT.search(text) is not None:
        text = RE_C_COMMENT.sub(' ', text)
    return text
