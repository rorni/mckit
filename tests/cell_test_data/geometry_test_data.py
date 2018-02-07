surface_data = {
    1: ('sx', [4, 2]),
    2: ('cx', [2]),
    3: ('px', [-3]),
    4: ('sx', [-3, 1]),
    5: ('px', [4]),
    6: ('sx', [4, 1]),
    7: ('cx', [3]),
    8: ('cx', [1]),
    9: ('px', [-5]),
    10: ('px', [8]),
}

polish_geoms = [
    [2, 'C', 3, 'I', 1, 'I', 5, 'C', 'I', 4, 'C', 'U'],
    [6, 'C', 1, 'C', 'I'],
    [6, 'C', 1, 'C', 'U'],
    [6, 1, 'I', 2, 'C', 'I', 5, 'C', 'I', 3, 'I', 4, 'C', 9, 'I', 'U', 3, 'C',
     4, 'C', 'I', 'U', 7, 'C', 'I', 10, 'C', 'I', 4, 'C', 10, 'C', 'I', 'U'],
    [5, 'C', 3, 'I', 2, 'C', 'I', 1, 'C', 'U'],
    [3, 8, 'C', 'I', 5, 'C', 'I', 4, 'C', 'U', 1, 'C', 'U', 2, 'C', 'I']
]

create_geom = [
    ('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}),
    ('I', {'negative': [1, 6]}),
    ('U', {'negative': [1, 6]}),
    ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]}),
    ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]}),
    ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})
]

simple_geoms = [
    [2, 'C', 3, 'I', 1, 'I', 5, 'C', 'I', 4, 'C', 'U'],
    [6, 'C'],
    [1, 'C'],
    [2, 'C', 3, 'I', 1, 'I', 5, 'C', 'I', 4, 'C', 'U'],
    [5, 'C', 3, 'I', 2, 'C', 'I', 1, 'C', 'U'],
    [3, 8, 'C', 'I', 5, 'C', 'I', 4, 'C', 'U', 1, 'C', 'U']
]

complement_geom = [
    ('I', {'negative': [('I', {'positive': [
        ('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})],
                               'negative': [5]})], 'positive': [4]}),
    ('U', {'positive': [1, 6]}),
    ('I', {'positive': [1, 6]}),
    ('I', {'negative': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3,4]})]})], 'negative': [7]})], 'negative': [10]})]}),
    ('I', {'negative': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'positive': [1]}),
    ('U', {'negative': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'positive': [2]})
]

intersection_geom = [
    [
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('I', {'negative': [1, 6]})]}),
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'negative': [1, 6]})]}),
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ],
    [
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('I', {'negative': [1, 6]})]}),
        ('I', {'positive': [('I', {'negative': [1, 6]}), ('U', {'negative': [1, 6]})]}),
        ('I', {'positive': [('I', {'negative': [1, 6]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('I', {'positive': [('I', {'negative': [1, 6]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('I', {'positive': [('I', {'negative': [1, 6]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ],
    [
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'negative': [1, 6]})]}),
        ('I', {'positive': [('I', {'negative': [1, 6]}), ('U', {'negative': [1, 6]})]}),
        ('I', {'positive': [('U', {'negative': [1, 6]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('I', {'positive': [('U', {'negative': [1, 6]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('I', {'positive': [('U', {'negative': [1, 6]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ],
    [
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('I', {'positive': [('I', {'negative': [1, 6]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('I', {'positive': [('U', {'negative': [1, 6]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('I', {'positive': [('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('I', {'positive': [('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ],
    [
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('I', {'positive': [('I', {'negative': [1, 6]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('I', {'positive': [('U', {'negative': [1, 6]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('I', {'positive': [('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ],
    [
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]}),
        ('I', {'positive': [('I', {'negative': [1, 6]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]}),
        ('I', {'positive': [('U', {'negative': [1, 6]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]}),
        ('I', {'positive': [('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [4, 3]})]})], 'negative': [7]})], 'negative': [10]})]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]}),
        ('I', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ]
]

union_geom = [
    [
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('I', {'negative': [1, 6]})]}),
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'negative': [1, 6]})]}),
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ],
    [
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('I', {'negative': [1, 6]})]}),
        ('U', {'positive': [('I', {'negative': [1, 6]}), ('U', {'negative': [1, 6]})]}),
        ('U', {'positive': [('I', {'negative': [1, 6]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('U', {'positive': [('I', {'negative': [1, 6]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('U', {'positive': [('I', {'negative': [1, 6]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ],
    [
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'negative': [1, 6]})]}),
        ('U', {'positive': [('I', {'negative': [1, 6]}), ('U', {'negative': [1, 6]})]}),
        ('U', {'positive': [('U', {'negative': [1, 6]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('U', {'positive': [('U', {'negative': [1, 6]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('U', {'positive': [('U', {'negative': [1, 6]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ],
    [
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('U', {'positive': [('I', {'negative': [1, 6]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('U', {'positive': [('U', {'negative': [1, 6]}), ('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]})]}),
        ('U', {'positive': [('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('U', {'positive': [('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ],
    [
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('U', {'positive': [('I', {'negative': [1, 6]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('U', {'positive': [('U', {'negative': [1, 6]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('U', {'positive': [('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]}), ('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]})]}),
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ],
    [
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [1, ('I', {'positive': [3], 'negative': [2]})]})], 'negative': [5]})], 'negative': [4]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]}),
        ('U', {'positive': [('I', {'negative': [1, 6]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]}),
        ('U', {'positive': [('U', {'negative': [1, 6]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]}),
        ('U', {'positive': [('U', {'positive': [('I', {'negative': [4, 10]}), ('I', {'positive': [('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [('I', {'positive': [1, 6]})], 'negative': [2]})], 'negative': [5]}), 3]}), ('I', {'positive': [9], 'negative': [4]})]}), ('I', {'negative': [3, 4]})]})], 'negative': [7]})], 'negative': [10]})]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]}),
        ('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [5]})], 'negative': [2]})], 'negative': [1]}), ('I', {'positive': [('U', {'positive': [('U', {'positive': [('I', {'positive': [('I', {'positive': [3], 'negative': [8]})], 'negative': [5]})], 'negative': [4]})], 'negative': [1]})], 'negative': [2]})]})
    ]
]

cell_complement_cases = [
    (0, 0), (-1, +1), (+1, -1), ([-1, 0, 1], [1, 0, -1])
]

cell_intersection_cases = [
    (-1, -1, -1), (-1, 0, -1), (-1, 1, -1), (0, -1, -1), (0, 0, 0), (0, 1, 0),
    (1, -1, -1), (1, 0, 0), (1, 1, 1),
    ([-1, -1, -1, 0, 0, 0, 1, 1, 1],
     [-1, 0, 1, -1, 0, 1, -1, 0, 1], [-1, -1, -1, -1, 0, 0, -1, 0, 1])
]

cell_union_cases = [
    (-1, -1, -1), (-1, 0, 0), (-1, 1, 1), (0, -1, 0), (0, 0, 0), (0, 1, 1),
    (1, -1, 1), (1, 0, 1), (1, 1, 1),
    ([-1, -1, -1, 0, 0, 0, 1, 1, 1],
     [-1, 0, 1, -1, 0, 1, -1, 0, 1], [-1, 0, 1, 0, 0, 1, 1, 1, 1])
]
