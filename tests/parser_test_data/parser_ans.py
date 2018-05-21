ans1 = {
    'title': 'mcnp parsing test file',
    'cells': {
        1: {
            'geometry': [1, 2, 'C', 'I', 3, 'I'],
            'IMPN': 1, 'MAT': 1, 'RHO': -2.0, 'name': 1
        },
        2: {
            'geometry': [1, 2, 'C', 3, 4, 'I', 'U', 'I'],
            'VOL': 1, 'MAT': 2, 'RHO': -3.5, 'name': 2
        },
        3: {
            'geometry': [2, 2, '#', 'I', 1, 'C', 3, 'U', 'C', 'I'],
            'IMPN': 1, 'IMPP': 1, 'name': 3
        },
        4: {
            'reference': 1, 'RHO': -3.0, 'name': 4
        }
    },
    'surfaces': {
        1: {'kind': 'SX', 'params': [4, 5], 'transform': 1, 'name': 1},
        2: {'kind': 'PX', 'params': [1], 'modifier': '*', 'name': 2},
        3: {'kind': 'S', 'params': [1, 2, -3, 4], 'name':3},
        4: {'kind': 'PY', 'params': [-5], 'name': 4}
    },
    'data': {
        'MODE': ['N', 'P'],
        'M': {
            1: {'atomic': [(1001, 0.1), (1002, 0.9)], 'name': 1},
            2: {'wgt': [(6012, 0.5, '50C'), (8016, 0.5, '21C')], 'name': 2},
            3: {'atomic': [(1001, 0.1), (1002, 0.9)], 'GAS': 1, 'name': 3},
            4: {'atomic': [(1001, 0.1)], 'wgt': [(1002, 0.9)], 'GAS': 1,
                'NLIB': '50C', 'name': 4}
        },
        'TR': {
            1: {'translation': [1, 2, 3], 'name': 1}
        }
    }
}

ans2 = {
    'title': 'mcnp parsing test file 2',
    'cells': {
        1: {
            'geometry': [1, 'C', 2, 'I', 3, 'C', 'U'], 'MAT': 1, 'RHO': -0.5,
            'IMPN': 1, 'name': 1
        },
        2: {
            'geometry': [1, 2, 'C', 3, 4, 'I', 5, 6, 'C', 'U', 'I', 'U', 'I',
                         7, 'U'],
            'MAT': 2, 'RHO': -1.0, 'U': 1, 'IMPN': 2, 'TRCL': 1, 'name': 2
        },
        3: {'geometry': [8, 9, 'I', 10, 'C', 'I'], 'FILL': {'universe': 1},
            'name': 3},
        4: {
            'geometry': [10, 11, 'C', 'I', 12, 'I'], 'name': 4,
            'TRCL': {'translation': [1, 2, 3],
                     'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
        },
        5: {
            'reference': 3, 'name': 5,
            'TRCL': {'translation': [1, 2, 3], 'indegrees': True,
                     'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
        },
        6: {
            'geometry': [16, 17, 'C', 'I', 18, 'I'],
            'FILL': {'universe': 1, 'transform': 2}, 'name': 6,
            'comment': ('$ comment1', '$ comment2', '$ comment3')
        },
        7: {
            'geometry': [19, 20, 'C', 'I', 21, 'I'], 'name': 7,
            'FILL': {'universe': 1, 'transform': {'translation': [1, 2, 3],
                     'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}}
        },
        8: {
            'geometry': [22, 23, 'C', 'I', 24, 'I'],
            'FILL': {'universe': 1, 'transform': {'translation': [1, 2, 3],
                     'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0],
                     'indegrees': True}}, 'name': 8
        }
    },
    'surfaces': {
        1: {'kind': 'PX', 'params': [1], 'modifier': '*', 'name': 1},
        2: {'kind': 'PY', 'params': [2], 'modifier': '+', 'name': 2},
        3: {'kind': 'PZ', 'params': [3], 'name': 3},
        4: {'kind': 'P', 'params': [1, 2, -3, -5], 'name': 4, 'comment': ('$ comment4',)},
        5: {'kind': 'SO', 'params': [3], 'name': 5},
        6: {'kind': 'SX', 'params': [4, 5], 'name': 6},
        7: {'kind': 'SY', 'params': [-4, 5], 'name': 7},
        8: {'kind': 'SZ', 'params': [2.0, 6.3], 'name': 8},
        9: {'kind': 'S', 'params': [-1, 2.3, -4.1, 6], 'name': 9},
        10: {'kind': 'CX', 'params': [2, 5], 'name': 10},
        11: {'kind': 'CY', 'params': [2, 5], 'name': 11},
        12: {'kind': 'CZ', 'params': [2, 5], 'name': 12},
        13: {'kind': 'C/X', 'params': [2, 3, 5], 'name': 13},
        14: {'kind': 'C/Y', 'params': [2, 3, 5], 'name': 14},
        15: {'kind': 'C/Z', 'params': [2, 3, 5], 'name': 15},
        16: {'kind': 'KX', 'params': [2, 0.5], 'name': 16},
        17: {'kind': 'KY', 'params': [2, 0.5], 'name': 17},
        18: {'kind': 'KZ', 'params': [2, 0.5], 'name': 18},
        19: {'kind': 'K/X', 'params': [1, 2, 3, 0.5], 'name': 19},
        20: {'kind': 'K/Y', 'params': [1, 2, 3, 0.5], 'name': 20},
        21: {'kind': 'K/Z', 'params': [1, 2, 3, 0.5], 'name': 21},
        22: {'kind': 'TX', 'params': [1, 2, 3, 4, 5, 8], 'name': 22},
        23: {'kind': 'TY', 'params': [1, 2, 3, 4, 5, 8], 'name': 23},
        24: {'kind': 'TZ', 'params': [1, 2, 3, 4, 5, 8], 'name': 24},
        25: {'kind': 'SQ', 'params': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'name': 25},
        26: {'kind': 'GQ', 'params': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'name': 26,
             'comment': ('$ comment5',)}
    },
    'data': {
        'TR': {
            1: {'translation': [1, 2, 3], 'name': 1, 'comment': ('$ comment6',),
                'rotation': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'inverted': True},
            2: {'translation': [1, 2, 3], 'indegrees': True, 'inverted': True,
                'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0], 'name': 2},
            3: {'translation': [1, 2, 3], 'indegrees': True, 'name': 3,
                'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]},
            4: {'translation': [1, 2, 3], 'indegrees': True, 'name': 4,
                'rotation': [30, 60, 90, 120, 30, 90]},
            5: {'translation': [1, 2, 3], 'indegrees': True, 'name': 5,
                'rotation': [30, 60, 90, 120, 30]},
            6: {'translation': [1, 2, 3], 'indegrees': True, 'name': 6,
                'rotation': [30, 60, 90]},
            7: {'translation': [1, 2, 3], 'indegrees': True, 'name': 7}
        }
    }
}

ans = {'parser1': ans1, 'parser2': ans2}
