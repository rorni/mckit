import pytest

from mckit.material import Element, Composition, Material


class TestElement:
    cases = [
        ('H', {}), ('1000', {}), ('1001', {}),
        ('CA', {}), ('ca', {}), ('Ca', {'lib': '21c'}),
        ('CA40', {'lib': '21c'}), ('CA-40', {}), ('Ca42', {'lib': '21c'}),
        ('ca-43', {}), ('CA-41', {}),
        ('U', {}), ('U', {'isomer': 1}), ('U235', {}),
        ('u235', {'isomer': 1, 'lib': '50c'}),
        ('U-238', {'comment': 'pure 238'}), ('92238', {}), ('92000', {}),
        ('Be', {}), ('Be-9', {}), ('4000', {}), ('4009', {}), (4000, {}),
        (4009, {})
    ]

    @pytest.mark.parametrize("case_no, expected", zip(range(len(cases)), [
        {'charge': 1, 'mass_number': 0, 'molar_mass': 1.0079, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 1, 'mass_number': 0, 'molar_mass': 1.0079, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 1, 'mass_number': 1, 'molar_mass': 1.007825, 'lib': None,
         'isomer': 0, 'comment': None},

        {'charge': 20, 'mass_number': 0, 'molar_mass': 40.078, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 20, 'mass_number': 0, 'molar_mass': 40.078, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 20, 'mass_number': 0, 'molar_mass': 40.078, 'lib': '21c',
         'isomer': 0, 'comment': None},
        {'charge': 20, 'mass_number': 40, 'molar_mass': 39.962591, 'lib': '21c',
         'isomer': 0, 'comment': None},
        {'charge': 20, 'mass_number': 40, 'molar_mass': 39.962591, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 20, 'mass_number': 42, 'molar_mass': 41.958618, 'lib': '21c',
         'isomer': 0, 'comment': None},
        {'charge': 20, 'mass_number': 43, 'molar_mass': 42.958767, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 20, 'mass_number': 41, 'molar_mass': 41, 'lib': None,
         'isomer': 0, 'comment': None},

        {'charge': 92, 'mass_number': 0, 'molar_mass': 238.0289, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 92, 'mass_number': 0, 'molar_mass': 238.0289, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 92, 'mass_number': 235, 'molar_mass': 235.043923,
         'lib': None, 'isomer': 0, 'comment': None},
        {'charge': 92, 'mass_number': 235, 'molar_mass': 235.043923,
         'lib': '50c', 'isomer': 1, 'comment': None},
        {'charge': 92, 'mass_number': 238, 'molar_mass': 238.050783,
         'lib': None, 'isomer': 0, 'comment': 'pure 238'},
        {'charge': 92, 'mass_number': 238, 'molar_mass': 238.050783,
         'lib': None, 'isomer': 0, 'comment': None},
        {'charge': 92, 'mass_number': 0, 'molar_mass': 238.0289, 'lib': None,
         'isomer': 0, 'comment': None},

        {'charge': 4, 'mass_number': 0, 'molar_mass': 9.012182, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 4, 'mass_number': 9, 'molar_mass': 9.012182, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 4, 'mass_number': 0, 'molar_mass': 9.012182, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 4, 'mass_number': 9, 'molar_mass': 9.012182, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 4, 'mass_number': 0, 'molar_mass': 9.012182, 'lib': None,
         'isomer': 0, 'comment': None},
        {'charge': 4, 'mass_number': 9, 'molar_mass': 9.012182, 'lib': None,
         'isomer': 0, 'comment': None}
    ]))
    def test_creation(self, case_no, expected):
        name, options = self.cases[case_no]
        elem = Element(name, **options)
        assert elem.charge == expected['charge']
        assert elem.mass_number == expected['mass_number']
        assert elem.lib == expected['lib']
        assert elem.isomer == expected['isomer']
        assert elem._comment == expected['comment']
        assert elem.molar_mass == pytest.approx(expected['molar_mass'], 1.e-4)

    @pytest.mark.parametrize("case_no, expected", zip(range(len(cases)), [
        [('H1', {}, 0.999885), ('H2', {}, 0.000115)],
        [('H1', {}, 0.999885), ('H2', {}, 0.000115)],
        [('H1', {}, 1.0)],

        [('CA40', {}, 0.96941), ('CA42', {}, 0.00647), ('CA43', {}, 0.00135),
         ('CA44', {}, 0.02086), ('CA46', {}, 0.00004), ('CA48', {}, 0.00187)],
        [('CA40', {}, 0.96941), ('CA42', {}, 0.00647), ('CA43', {}, 0.00135),
         ('CA44', {}, 0.02086), ('CA46', {}, 0.00004), ('CA48', {}, 0.00187)],
        [('CA40', {'lib': '21c'}, 0.96941), ('CA42', {'lib': '21c'}, 0.00647),
         ('CA43', {'lib': '21c'}, 0.00135), ('CA44', {'lib': '21c'}, 0.02086),
         ('CA46', {'lib': '21c'}, 0.00004), ('CA48', {'lib': '21c'}, 0.00187)],
        [('CA40', {'lib': '21c'}, 1.0)],
        [('CA40', {}, 1.0)],
        [('CA42', {'lib': '21c'}, 1.0)],
        [('CA43', {}, 1.0)],
        [],

        [('U234', {}, 0.000055), ('U235', {}, 0.007200),
         ('U238', {}, 0.992745)],
        [('U234', {}, 0.000055), ('U235', {}, 0.007200),
         ('U238', {}, 0.992745)],
        [('U235', {}, 1.0)],
        [('U235', {'isomer': 1, 'lib': '50c'}, 1.0)],
        [('U238', {}, 1.0)],
        [('U238', {}, 1.0)],
        [('U234', {}, 0.000055), ('U235', {}, 0.007200),
         ('U238', {}, 0.992745)],

        [('BE9', {}, 1.0)],
        [('BE9', {}, 1.0)],
        [('BE9', {}, 1.0)],
        [('BE9', {}, 1.0)],
        [('BE9', {}, 1.0)],
        [('BE9', {}, 1.0)]
    ]))
    def test_expand(self, case_no, expected):
        name, options = self.cases[case_no]
        elem = Element(name, **options)
        expanded_ans = {Element(nam, **opt): pytest.approx(frac, rel=1.e-5)
                        for nam, opt, frac in expected}
        expanded = elem.expand()
        assert expanded == expanded_ans

    @pytest.mark.parametrize("case_no, expected", zip(range(len(cases)), [
        'H', 'H', 'H-1', 'Ca', 'Ca', 'Ca', 'Ca-40', 'Ca-40', 'Ca-42', 'Ca-43',
        'Ca-41', 'U', 'U', 'U-235', 'U-235m', 'U-238', 'U-238', 'U', 'Be',
        'Be-9', 'Be', 'Be-9', 'Be', 'Be-9'
    ]))
    def test_str(self, case_no, expected):
        name, options = self.cases[case_no]
        elem = Element(name, **options)
        assert expected == str(elem)

    @pytest.mark.parametrize("case_no, expected", zip(range(len(cases)), [
        '1000', '1000', '1001', '20000', '20000', '20000.21c', '20040.21c',
        '20040', '20042.21c', '20043', '20041', '92000', '92000', '92235',
        '92235.50c', '92238', '92238', '92000', '4000', '4009', '4000', '4009',
        '4000', '4009'
    ]))
    def test_mcnp_repr(self, case_no, expected):
        name, options = self.cases[case_no]
        elem = Element(name, **options)
        assert expected == elem.mcnp_repr()

    @pytest.mark.parametrize("case_no, expected", zip(range(len(cases)), [
        'H', 'H', 'H1', 'Ca', 'Ca', 'Ca', 'Ca40', 'Ca40', 'Ca42', 'Ca43',
        'Ca41', 'U', 'U', 'U235', 'U235m', 'U238', 'U238', 'U', 'Be', 'Be9',
        'Be', 'Be9', 'Be', 'Be9'
    ]))
    def test_fispact_repr(self, case_no, expected):
        name, options = self.cases[case_no]
        elem = Element(name, **options)
        assert expected == elem.fispact_repr()

    equality = [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1]
    ]

    @pytest.mark.parametrize('arg1', range(len(cases)))
    @pytest.mark.parametrize('arg2', range(len(cases)))
    def test_eq(self, arg1, arg2):
        name1, options1 = self.cases[arg1]
        name2, options2 = self.cases[arg2]
        elem1 = Element(name1, **options1)
        elem2 = Element(name2, **options2)
        test_result = (elem1 == elem2)
        assert test_result == bool(self.equality[arg1][arg2])
