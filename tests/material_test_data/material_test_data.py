creation_cases = [
    {'weight': [('N', 0.755465), ('O', 0.23148), ('AR', 0.012886)], 'density': 1.2929e-3, 'name': 1, 'lib': '21c'},
    {'atomic': [('N', 0.78479)], 'weight': [('O', 0.23148), ('AR', 0.012886)], 'density': 1.2929e-3, 'name': 1, 'lib': '21c'},
    {'atomic': [('N', 0.78479), ('O', 0.21052)], 'weight': [('AR', 0.012886)], 'concentration': 5.3509e+19, 'name': 1, 'lib': '21c'},
    {'atomic': [('N', 0.78479), ('Ar', 0.0046936)], 'weight': [('O', 0.23148)], 'concentration': 5.3509e+19, 'name': 1, 'lib': '21c'},
    {'atomic': [('N', 1)], 'density': 1.251e-3},
    {'atomic': [('O', 1)], 'density': 1.42897e-3},
    {'atomic': [('Ar', 1)], 'density': 1.784e-3}

]


failure_cases = [
    {},
    {'atomic': [('N', 1)]},
    {'weight': [('N', 1)]},
    {'atomic': [('N', 1)], 'weight': [('N', 1)]},
    {'composition': {'atomic': [('N', 1)]}},
    {'composition': {'atomic': [('N', 1)]}, 'atomic': [('N', 1)]},
    {'composition': {'atomic': [('N', 1)]}, 'weight': [('N', 1)]},
    {'composition': {'atomic': [('N', 1)]}, 'atomic': [('N', 1)], 'weight': [('N', 1)]},
    {'density': 7.8},
    {'concentration': 1.e+23},
    {'density': 7.8, 'concentration': 1.e+23},
    {'density': 7.8, 'concentration': 1.e+23, 'atomic': [('N', 1)]},
    {'density': 7.8, 'concentration': 1.e+23, 'weight': [('N', 1)]},
    {'density': 7.8, 'concentration': 1.e+23, 'atomic': [('N', 1)], 'weight': [('N', 1)]},
    {'density': 7.8, 'composition': {'atomic': [('N', 1)]}, 'atomic': [('N', 1)]},
    {'density': 7.8, 'composition': {'atomic': [('N', 1)]}, 'weight': [('N', 1)]},
    {'density': 7.8, 'composition': {'atomic': [('N', 1)]}, 'atomic': [('N', 1)], 'weight': [('N', 1)]},
    {'concentration': 7.8, 'composition': {'atomic': [('N', 1)]}, 'atomic': [('N', 1)]},
    {'concentration': 7.8, 'composition': {'atomic': [('N', 1)]}, 'weight': [('N', 1)]},
    {'concentration': 7.8, 'composition': {'atomic': [('N', 1)]}, 'atomic': [('N', 1)], 'weight': [('N', 1)]}
]

equal_cases = [
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
]

correct_cases = [
    {'new_vol': 5, 'old_vol': 2.5},
    {'new_vol': 4, 'old_vol': 6},
    {'factor': 2}
]

mixture_cases = [
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 1)], 'volume'),
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 0.8)], 'volume'),
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 1.2)], 'volume'),
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 1)], 'weight'),
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 0.8)], 'weight'),
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 1.2)], 'weight'),
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 1)], 'atomic'),
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 0.8)], 'atomic'),
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 1.2)], 'atomic'),
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 0.78084),
      ({'atomic': [('O', 1)], 'density': 1.42897e-3}, 0.20948),
      ({'atomic': [('Ar', 1)], 'density': 1.784e-3}, 0.00934)], 'volume'),
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 0.755465),
      ({'atomic': [('O', 1)], 'density': 1.42897e-3}, 0.23148),
      ({'atomic': [('Ar', 1)], 'density': 1.784e-3}, 0.012886)], 'weight'),
    ([({'atomic': [('N', 1)], 'density': 1.251e-3}, 0.78479),
      ({'atomic': [('O', 1)], 'density': 1.42897e-3}, 0.21052),
      ({'atomic': [('Ar', 1)], 'density': 1.784e-3}, 0.0046936)], 'atomic')
]

mixture_answers = [
    {'atomic': [('N', 1)], 'density': 1.251e-3},
    {'atomic': [('N', 1)], 'density': 1.251e-3 * 0.8},
    {'atomic': [('N', 1)], 'density': 1.251e-3 * 1.2},
    {'atomic': [('N', 1)], 'density': 1.251e-3},
    {'atomic': [('N', 1)], 'density': 1.251e-3},
    {'atomic': [('N', 1)], 'density': 1.251e-3},
    {'atomic': [('N', 1)], 'density': 1.251e-3},
    {'atomic': [('N', 1)], 'density': 1.251e-3},
    {'atomic': [('N', 1)], 'density': 1.251e-3},
    {'weight': [('N', 0.755465), ('O', 0.23148), ('AR', 0.012886)], 'density': 1.2929e-3},
    {'weight': [('N', 0.755465), ('O', 0.23148), ('AR', 0.012886)], 'density': 1.2929e-3},
    {'weight': [('N', 0.755465), ('O', 0.23148), ('AR', 0.012886)], 'density': 1.2929e-3}
]