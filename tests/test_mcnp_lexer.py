import unittest

from mckit.parser import lexer
from .parser_test_data import lex_ans


class TestLexer(unittest.TestCase):
    def test_lex(self):
        for i, name in enumerate(lex_ans.ans.keys()):
            lexer.begin('INITIAL')
            with (self.subTest(i=i)):
                with open('tests/parser_test_data/{0}.txt'.format(name)) as f:
                    text = f.read()
                lexer.input(text)
                for l in lex_ans.ans[name]:
                    t = lexer.token()
                    #print(t.lineno, t.value)
                    self.assertEqual(l[0], t.type)
                    self.assertEqual(l[1], t.value)
                    self.assertEqual(l[2], t.lineno)


if __name__ == '__main__':
    unittest.main()
