import unittest

from mckit.parser import lexer, parser
from .parser_test_data import lex_ans
from .parser_test_data import parser_ans


class TestLexer(unittest.TestCase):
    def test_lex(self):
        for i, name in enumerate(lex_ans.ans.keys()):
            lexer.begin('INITIAL')
            with (self.subTest(i=i)):
                with open('tests/parser_test_data/{0}.txt'.format(name)) as f:
                    text = f.read()
                lexer.input(text)
                for l in lex_ans.ans[name]:
                    if isinstance(l[0], str):
                        t = lexer.token()
                        #print(t.lineno, t.value)
                        self.assertEqual(l[0], t.type)
                        self.assertEqual(l[1], t.value)
                        self.assertEqual(l[2], t.lineno)
                    else:
                        with self.assertRaises(ValueError) as ex:
                            lexer.token()
                        msg, sym, line, column = ex.exception.args
                        self.assertEqual(l[1], line)
                        self.assertEqual(l[2], column)


class TestParser(unittest.TestCase):
    def test_parse(self):
        for i, name in enumerate(parser_ans.ans.keys()):
            with (self.subTest(i=i)):
                with open('tests/parser_test_data/{0}.txt'.format(name)) as f:
                    text = f.read()
                lexer.begin('INITIAL')
                title, cells, surfaces, data = parser.parse(text)
                ans = parser_ans.ans[name]
                self.assertEqual(title, ans['title'])
                self.recursive_dict_equality(cells, ans['cells'])
                self.recursive_dict_equality(surfaces, ans['surfaces'])
                self.recursive_dict_equality(data, ans['data'])

    def recursive_dict_equality(self, data1, data2):
        self.assertCountEqual(data1.keys(), data2.keys())
        for key, val1 in data1.items():
            val2 = data2[key]
            if isinstance(val1, list) or isinstance(val1, tuple):
                self.assertIsInstance(val2, val1.__class__)
                self.recursive_list_equality(val1, val2)
            elif isinstance(val1, dict):
                self.assertIsInstance(val2, dict)
                self.recursive_dict_equality(val1, val2)
            else:
                self.assertEqual(val1, val2)

    def recursive_list_equality(self, data1, data2):
        self.assertCountEqual(data1, data2)
        for d1, d2 in zip(data1, data2):
            if isinstance(d1, list) or isinstance(d1, tuple):
                self.assertIsInstance(d2, d1.__class__)
                self.recursive_list_equality(d1, d2)
            elif isinstance(d1, dict):
                self.assertIsInstance(d2, dict)
                self.recursive_dict_equality(d1, d2)
            else:
                self.assertEqual(d1, d2)


if __name__ == '__main__':
    unittest.main()
