ans1 = {
    'title': 'mcnp parsing test file',
    'cells': {
        1: {
            'geometry': [1, 2, 'C', 'I', 3, 'I'],
            ('IMP', 'N') : 1, 'MAT': 1, 'RHO': -2.0
        },
        2: {
            'geometry': [1, 2, 'C', 3, 4, 'I', 'U', 'I'],
            'VOL': 1, 'MAT': 2, 'RHO': -3.5
        },
        3: {
            'geometry': [2, 2, '#', 'I', 1, 'C', 3, 'U', 'C', 'I'],
            ('IMP', 'N'): 1, ('IMP', 'P'): 1
        },
        4: {
            'reference': 1, 'RHO': -3.0
        }
    },
    'surfaces': {
        1: ('SX', [4, 5], {'transform': 1}),
        2: ('PX', [1], {'modifier': '*'}),
        3: ('S', [1, 2, -3, 4], {}),
        4: ('PY', [-5], {})
    },
    'data': {
        'MODE': ['N', 'P'],
        'M': {
            1: {'atomic': [(1001, 0.1), (1002, 0.9)]},
            2: {'wgt': [(6012, 0.5, 50, 'c'), (8016, 0.5, 21, 'c')]}
        },
        'TR': {
            1: {'translation' : [1, 2, 3]}
        }
    }
}

ans = {'parser1': ans1}
