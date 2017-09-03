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
            2: {'wgt': [(6012, 0.5, '50C'), (8016, 0.5, '21C')]},
            3: {'atomic': [(1001, 0.1), (1002, 0.9)], 'GAS': 1},
            4: {'atomic': [(1001, 0.1)], 'wgt': [(1002, 0.9)], 'GAS': 1, 'NLIB': '50C'}
        },
        'TR': {
            1: {'translation' : [1, 2, 3]}
        }
    }
}

ans2 = {
    'title': 'mcnp parsing test file 2',
    'cells': {
        1: {
            'geometry': [1, 'C', 2, 'I', 3, 'C', 'U'], 'MAT': 1, 'RHO': -0.5,
            ('IMP', 'N') : 1
        },
        2: {
            'geometry': [1, 2, 'C', 3, 4, 'I', 5, 6, 'C', 'U', 'I', 'U', 'I',
                         7, 'U'],
            'MAT': 2, 'RHO': -1.0, 'U': 1, ('IMP', 'N'): 2, 'TRCL': 1
        },
        3: {'geometry': [8, 9, 'I', 10, 'C', 'I'], 'FILL': {'universe': 1}},
        4: {
            'geometry': [10, 11, 'C', 'I', 12, 'I'],
            'TRCL': {'translation': [1, 2, 3],
                     'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
        },
        5: {
            'reference': 3,
            'TRCL': {'translation': [1, 2, 3], 'indegrees': True,
                     'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
        },
        6: {
            'geometry': [16, 17, 'C', 'I', 18, 'I'],
            'FILL': {'universe': 1, 'transform': 2}
        },
        7: {
            'geometry': [19, 20, 'C', 'I', 21, 'I'],
            'FILL': {'universe': 1, 'transform': {'translation': [1, 2, 3],
                     'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}}
        },
        8: {
            'geometry': [22, 23, 'C', 'I', 24, 'I'],
            'FILL': {'universe': 1, 'transform': {'translation': [1, 2, 3],
                     'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0],
                     'indegrees': True}}
        }
    },
    'surfaces': {
        1: ('PX', [1], {'modifier': '*'}),
        2: ('PY', [2], {'modifier': '+'}),
        3: ('PZ', [3], {}), 4: ('P', [1, 2, -3, -5], {}), 5: ('SO', [3], {}),
        6: ('SX', [4, 5], {}), 7: ('SY', [-4, 5], {}), 8: ('SZ', [2.0, 6.3], {}),
        9: ('S', [-1, 2.3, -4.1, 6], {}), 10: ('CX', [2, 5], {}),
        11: ('CY', [2, 5], {}), 12: ('CZ', [2, 5], {}),
        13: ('C/X', [2, 3, 5], {}), 14: ('C/Y', [2, 3, 5], {}),
        15: ('C/Z', [2, 3, 5], {}), 16: ('KX', [2, 0.5], {}),
        17: ('KY', [2, 0.5], {}), 18: ('KZ', [2, 0.5], {}),
        19: ('K/X', [1, 2, 3, 0.5], {}), 20: ('K/Y', [1, 2, 3, 0.5], {}),
        21: ('K/Z', [1, 2, 3, 0.5], {}), 22: ('TX', [1, 2, 3, 4, 5, 8], {}),
        23: ('TY', [1, 2, 3, 4, 5, 8], {}),
        24: ('TZ', [1, 2, 3, 4, 5, 8], {}),
        25: ('SQ', [1, 2, 3, 4, 5, 6, 7, 8, 9], {}),
        26: ('GQ', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], {})
    },
    'data': {
        'TR': {
            1: {'translation': [1, 2, 3], 'rotation': [1, 1, 1, 2, 2, 2, 3, 3, 3],
                'inverted': True},
            2: {'translation': [1, 2, 3], 'indegrees': True, 'inverted': True,
                'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]},
            3: {'translation': [1, 2, 3], 'indegrees': True,
                'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]},
            4: {'translation': [1, 2, 3], 'indegrees': True,
                'rotation': [30, 60, 90, 120, 30, 90]},
            5: {'translation': [1, 2, 3], 'indegrees': True,
                'rotation': [30, 60, 90, 120, 30]},
            6: {'translation': [1, 2, 3], 'indegrees': True, 'rotation': [30, 60, 90]},
            7: {'translation': [1, 2, 3], 'indegrees': True}
        }
    }
}

ans = {'parser1': ans1, 'parser2': ans2}
