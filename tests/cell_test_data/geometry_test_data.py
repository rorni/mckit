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
    [3, 8, 'C', 'I', 5, 'C', 'I', 4, 'C', 'U', 6, 'C', 'U', 2, 'C', 'I']
]

create_geom = [
    ('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]),
    ('I', [('C', [6]), ('C', [1])]),
    ('U', [('C', [6]), ('C', [1])]),
    ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])]),
    ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])]),
    ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])
]

node_complexity_data = [5, 2, 2, 13, 4, 6]

simple_geoms = [
    [2, 'C', 3, 'I', 1, 'I', 5, 'C', 'I', 4, 'C', 'U'],
    [6, 'C'],
    [1, 'C'],
    [2, 'C', 3, 'I', 1, 'I', 5, 'C', 'I', 4, 'C', 'U'],
    [5, 'C', 3, 'I', 2, 'C', 'I', 1, 'C', 'U'],
    [3, 8, 'C', 'I', 5, 'C', 'I', 4, 'C', 'U', 6, 'C', 'U']
]

complement_geom = [
    ('I', [('U', [('E', [2]), ('C', [3]), ('C', [1]), ('E', [5])]), ('E', [4])]),
    ('U', [('E', [6]), ('E', [1])]),
    ('I', [('E', [6]), ('E', [1])]),
    ('I', [('U', [('E', [4]), ('E', [10])]), ('U', [('E', [7]), ('E', [10]), ('I', [('U', [('E', [3]), ('E', [4])]), ('U', [('E', [4]), ('C', [9])]), ('U', [('C', [6]), ('C', [1]), ('E', [2]), ('E', [5]), ('C', [3])])])])]),
    ('I', [('E', [1]), ('U', [('E', [5]), ('C', [3]), ('E', [2])])]),
    ('U', [('E', [2]), ('I', [('E', [6]), ('E', [4]), ('U', [('C', [3]), ('E', [8]), ('E', [5])])])])
]

intersection_geom = [
    [
        ('I', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [6]), ('C', [1])])]),
        ('I', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [6]), ('C', [1])])]),
        ('I', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('I', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('I', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ],
    [
        ('I', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [6]), ('C', [1])])]),
        ('I', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [6]), ('C', [1])])]),
        ('I', [('I', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('I', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('I', [('I', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ],
    [
        ('I', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [6]), ('C', [1])])]),
        ('I', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [6]), ('C', [1])])]),
        ('I', [('U', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('I', [('U', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('I', [('U', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ],
    [
        ('I', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('I', [('I', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('I', [('U', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('I', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('I', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ],
    [
        ('I', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('I', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('I', [('U', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('I', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('I', [('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ],
    [
        ('I', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])]),
        ('I', [('I', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])]),
        ('I', [('U', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])]),
        ('I', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])]),
        ('I', [('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ]
]

union_geom = [
    [
        ('U', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [6]), ('C', [1])])]),
        ('U', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [6]), ('C', [1])])]),
        ('U', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('U', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('U', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ],
    [
        ('U', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [6]), ('C', [1])])]),
        ('U', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [6]), ('C', [1])])]),
        ('U', [('I', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('U', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('U', [('I', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ],
    [
        ('U', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [6]), ('C', [1])])]),
        ('U', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [6]), ('C', [1])])]),
        ('U', [('U', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('U', [('U', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('U', [('U', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ],
    [
        ('U', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('U', [('I', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('U', [('U', [('C', [6]), ('C', [1])]), ('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])])]),
        ('U', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('U', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ],
    [
        ('U', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('U', [('I', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('U', [('U', [('C', [6]), ('C', [1])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('U', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])]), ('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])])]),
        ('U', [('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ],
    [
        ('U', [('U', [('I', [('C', [2]), ('E', [3]), ('E', [1]), ('C', [5])]), ('C', [4])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])]),
        ('U', [('I', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])]),
        ('U', [('U', [('C', [6]), ('C', [1])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])]),
        ('U', [('U', [('I', [('C', [4]), ('C', [10])]), ('I', [('C', [7]), ('C', [10]), ('U', [('I', [('C', [3]), ('C', [4])]), ('I', [('C', [4]), ('E', [9])]), ('I', [('E', [6]), ('E', [1]), ('C', [2]), ('C', [5]), ('E', [3])])])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])]),
        ('U', [('U', [('C', [1]), ('I', [('C', [5]), ('E', [3]), ('C', [2])])]), ('I', [('C', [2]), ('U', [('C', [6]), ('C', [4]), ('I', [('E', [3]), ('C', [8]), ('C', [5])])])])])
    ]
]

node_points = [
    [-6, 0, 0], [-3.5, 0, 0], [-3.5, 1.5, 0], [-2.5, 1.5, 0], [-1, 2.5, 0], [1, -1.5, 0], [1, -0.5, 0], [2.5, 0.5, 0],
    [4, -0.5, 0], [5.5, 0.5, 0], [7, -0.5, 0]
]

node_test_point_ans = [
    [-1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, +1, +1, +1, -1],
    [-1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1],
    [-1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1],
    [-1, +1, -1, -1, -1, -1, +1, +1, +1, -1, -1]
]

node_boxes_data = [
    {'base': [0, 0, 0], 'ex': [2.5, 0, 0], 'ey': [0, 3.5, 0], 'ez': [0, 0, 3]},
    {'base': [-2, 0, 0], 'ex': [-2.5, 0, 0], 'ey': [0, -3, 0], 'ez': [0, 0, 3]},
    {'base': [4.5, 0, 0], 'ex': [2, 0, 0], 'ey': [0, -1.5, 0], 'ez': [0, 0, 1.5]}
]

node_box_ans = [
    [
        (0, [('U', [('I', [('C', [2]), ('E', [1])])])]),
        (-1, [('I', [('C', [6])])]),
        (0, [('U', [('C', [1])])]),
        (0, [('U', [('I', [('C', [7]), ('U', [('I', [('E', [1]), ('C', [2])])])])])]),
        (0, [('U', [('C', [1]), ('I', [('C', [2])])])]),
        (0, [('I', [('C', [2]), ('U', [('I', [('C', [8])])])])])
    ],
    [
        (0, [('U', [('I', [('C', [2]), ('E', [3])]), ('C', [4])])]),
        (-1, [('I', [('C', [6])]), ('I', [('C', [1])])]),
        (-1, [('U', [('C', [6]), ('C', [1])])]),
        (0, [('U', [('I', [('C', [4])]), ('I', [('C', [7]), ('U', [('I', [('C', [4]), ('C', [3])]), ('I', [('C', [4])]), ('I', [('E', [3]), ('C', [2])])])])])]),
        (0, [('U', [('I', [('C', [2]), ('E', [3])])])]),
        (0, [('I', [('C', [2]), ('U', [('C', [4]), ('I', [('C', [8]), ('E', [3])])])])])
    ],
    [
        (-1, [('U', [('C', [4]), ('I', [('C', [5])])])]),
        (0, [('I', [('C', [6]), ('C', [1])])]),
        (0, [('U', [('C', [6]), ('C', [1])])]),
        (-1, [('U', [('I', [('C', [4])]), ('I', [('U', [('I', [('C', [4])]), ('I', [('C', [4])]), ('I', [('C', [5])])])])]),
              ('U', [('I', [('C', [4])]), ('I', [('U', [('I', [('C', [3])]), ('I', [('C', [4])]), ('I', [('C', [5])])])])])]),
        (0, [('U', [('C', [1])])]),
        (0, [('I', [('C', [2]), ('U', [('C', [6])])])])
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
