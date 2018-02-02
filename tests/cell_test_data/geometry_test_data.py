surface_data = {
    1: ('so', [1]),
    2: ('so', [2]),
    3: ('so', [3]),
    4: ('py', [1]),
    5: ('py', [-1]),
    6: ('c/z', [5, 0, 1]),
    7: ('c/z', [5, 0, 2]),
    8: ('pz', [-2]),
    9: ('pz', [2]),
    10: ('px', [5]),
    11: ('px', [-1.5]),
    12: ('px', [1.6]),
    13: ('py', [3.6])
}

terms = [
    {'positive': {6}, 'negative': {6, 7}},  # Empty set
    {'positive': {5, 8}, 'negative': {3, 9}},
    {'positive': {5, 8, 11}, 'negative': {3, 9, 13}},
    {'negative': {2, 3}},
    {'positive': {12, 8}, 'negative': {9, 7, 10}},
    {'positive': {8}, 'negative': {7, 9}},
    {'negative': {7}},
    {'negative': {6}}
]

additives = [
    [0, 1, 2], [1, 2, 4, 5], [3, 6], [6, 7, 1], [1, 2, 3, 4]
]

ag_polish_data = [
    (
        [2], [{'positive': {2}}]
    ),
    (
        [2, 'C'], [{'negative': {2}}]
    ),
    (
        [2, 'C', 3, 'C', 'I', 5, 'I'], [{'positive': {5}, 'negative': {2, 3}}]
    ),
    (
        [2, 'C', 3, 'C', 'I', 7, 'U'], [{'positive': {7}}, {'negative': {2, 3}}]
    ),
    (
        [2, 3, 'U', 5, 'I'], [{'positive': {2, 5}}, {'positive': {3, 5}}]
    ),
    (
        [2, 3, 'C', 'I', 'C'], [{'negative': {2}}, {'positive': {3}}]
    )
]

ag_create = [
    [{'positive': {5, 8}, 'negative': {3, 9}}],
    [{'positive': {5, 8}, 'negative': {3, 9}},
     {'positive': {8}, 'negative': {7, 9}}],
    [{'negative': {2, 3}}, {'negative': {7}}],
    [{'negative': {7}}, {'negative': {6}},
     {'positive': {5, 8}, 'negative': {3, 9}}],
    [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
     {'positive': {12, 8}, 'negative': {9, 7, 10}}]
]

ag_simplify = [
    [{'positive': {5, 8}, 'negative': {3, 9}}],
    [{'positive': {5, 8}, 'negative': {3, 9}},
     {'positive': {8}, 'negative': {7, 9}}],
    [{'negative': {2}}, {'negative': {7}}],
    [{'negative': {7}}, {'positive': {5, 8}, 'negative': {3, 9}}],
    [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2}},
     {'positive': {8}, 'negative': {9, 7, 10}}]
]

ag_box_data = [
    (
    {'base': [0, 0, -1], 'ex': [1.5, 0, 0], 'ey': [0, 1.5, 0], 'ez': [0, 0, 2]},
    [(+1, [[{}]]), (+1, [[{}]]), (0, [[{'negative': {2}}]]), (+1, [[{}]]),
     (+1, [[{}]])]
    ),
    (
    {'base': [2.5, 0, -1], 'ex': [3, 0, 0], 'ey': [0, 2.5, 0], 'ez': [0, 0, 4]},
    [(0, [[{'negative': {3, 9}}]]),
     (0, [[{'negative': {3, 9}}, {'negative': {7, 9}}]]),
     (0, [[{'negative': {7}}]]),
     (0, [[{'negative': {3, 9}}, {'negative': {7}}, {'negative': {6}}]]),
     (0, [[{'negative': {3, 9}}, {'negative': {7, 9, 10}}]])]
    )
]

ag_intersection2 = [
    [
        [{}], [{}], [{}], [{}], [{}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {2, 3, 9}},
         {'positive': {5, 8}, 'negative': {7, 3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}]
    ],
    [
        [{'positive': {5, 8, 11}, 'negative': {3, 9, 13}}],
        [{'positive': {5, 8, 11}, 'negative': {3, 9, 13}}],
        [{'positive': {5, 8, 11}, 'negative': {2, 3, 9, 13}},
         {'positive': {5, 8, 11}, 'negative': {7, 3, 9, 13}}],
        [{'positive': {5, 8, 11}, 'negative': {3, 9, 13}}],
        [{'positive': {5, 8, 11}, 'negative': {3, 9, 13}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {2, 3, 9}}],
        [{'positive': {5, 8}, 'negative': {2, 3, 9}},
         {'positive': {8}, 'negative': {2, 3, 7, 9}}],
        [{'negative': {2, 3}}],
        [{'negative': {2, 3, 7}}, {'negative': {2, 3, 6}},
         {'positive': {5, 8}, 'negative': {2, 3, 9}}],
        [{'negative': {2, 3}}]
    ],
    [
        [{'positive': {5, 8, 12}, 'negative': {3, 7, 9, 10}}],
        [{'positive': {8, 12}, 'negative': {7, 9, 10}}],
        [{'positive': {12, 8}, 'negative': {9, 7, 10}}],
        [{'positive': {12, 8}, 'negative': {9, 7, 10}}],
        [{'positive': {12, 8}, 'negative': {9, 7, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 7, 9}}],
        [{'positive': {8}, 'negative': {7, 9}}],
        [{'positive': {8}, 'negative': {7, 9}}],
        [{'positive': {8}, 'negative': {7, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 7, 9}},
         {'positive': {8}, 'negative': {2, 3, 7, 9}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 7, 9}}],
        [{'positive': {8}, 'negative': {7, 9}}],
        [{'negative': {7}}],
        [{'negative': {7}}],
        [{'positive': {5, 8}, 'negative': {3, 7, 9}}, {'negative': {2, 3, 7}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 6, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 6, 9}},
         {'positive': {8}, 'negative': {7, 6, 9}}],
        [{'negative': {2, 3, 6}}, {'negative': {6, 7}}],
        [{'negative': {6}}],
        [{'positive': {5, 8}, 'negative': {3, 6, 9}}, {'negative': {2, 3, 6}},
         {'positive': {12, 8}, 'negative': {9, 7, 6, 10}}]
    ]
]

ag_intersection1 = [
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {2, 3, 9}},
         {'positive': {5, 8}, 'negative': {3, 7, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'positive': {5, 8}, 'negative': {2, 3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {2, 3, 7, 9}},
         {'positive': {12, 8}, 'negative': {7, 9, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {2, 3, 9}},
         {'positive': {5, 8}, 'negative': {3, 7, 9}}],
        [{'positive': {5, 8}, 'negative': {2, 3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'negative': {2, 3}}, {'negative': {7}}],
        [{'negative': {2, 3, 6}}, {'negative': {7}},
         {'positive': {5, 8}, 'negative': {2, 3, 9}}],
        [{'negative': {2, 3}}, {'positive': {5, 8}, 'negative': {3, 7, 9}},
         {'positive': {12, 8}, 'negative': {7, 9, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'negative': {2, 3, 6}}, {'negative': {7}},
         {'positive': {5, 8}, 'negative': {2, 3, 9}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'negative': {2, 3, 6}}, {'negative': {2, 3, 7}},
         {'positive': {12, 8}, 'negative': {7, 9, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {2, 3, 7, 9}},
         {'positive': {12, 8}, 'negative': {7, 9, 10}}],
        [{'negative': {2, 3}}, {'positive': {5, 8}, 'negative': {3, 7, 9}},
         {'positive': {12, 8}, 'negative': {7, 9, 10}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'negative': {2, 3, 6}}, {'negative': {2, 3, 7}},
         {'positive': {12, 8}, 'negative': {7, 9, 10}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}]
    ]
]

ag_union2 = [
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'negative': {2, 3}}, {'negative': {7}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}, ],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'negative': {2, 3}}, {'negative': {7}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'negative': {2, 3}}, {'negative': {7}},
         {'positive': {5, 8, 11}, 'negative': {3, 9, 13}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}, {'negative': {2, 3}}],
        [{'negative': {2, 3}}, {'negative': {7}}],
        [{'negative': {7}}, {'negative': {6}}, {'negative': {2, 3}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'negative': {2, 3}}, {'negative': {7}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'negative': {2, 3}}, {'negative': {7}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {8}, 'negative': {7, 9}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {7}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {7}}],
        [{'negative': {2, 3}}, {'negative': {7}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'negative': {7}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {6}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}, {'negative': {6}}],
        [{'negative': {2, 3}}, {'negative': {7}}, {'negative': {6}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}, {'negative': {6}}]
    ]
]

ag_union1 = [
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'negative': {2, 3}}, {'negative': {7}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}},
         {'negative': {2, 3}}, {'negative': {7}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'negative': {7}}, {'negative': {6}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {8}, 'negative': {7, 9}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'negative': {2, 3}}, {'negative': {7}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'negative': {2, 3}}, {'negative': {7}}],
        [{'negative': {2, 3}}, {'negative': {7}}],
        [{'negative': {2, 3}}, {'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'negative': {2, 3}}, {'negative': {7}},
         {'positive': {5, 8}, 'negative': {3, 9}}]
    ],
    [
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'negative': {2, 3}}, {'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}}]
    ],
    [
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}],
        [{'positive': {5, 8}, 'negative': {3, 9}},
         {'positive': {8}, 'negative': {7, 9}}, {'negative': {2, 3}}],
        [{'negative': {2, 3}}, {'negative': {7}},
         {'positive': {5, 8}, 'negative': {3, 9}}],
        [{'negative': {7}}, {'negative': {6}},
         {'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}}],
        [{'positive': {5, 8}, 'negative': {3, 9}}, {'negative': {2, 3}},
         {'positive': {12, 8}, 'negative': {9, 7, 10}}]
    ]
]

ag_contains = [
    [True,  False, False, False, False],
    [True,  True,  False, False, False],
    [False, False, True,  False, False],
    [True,  True,  False, True,  False],
    [True,  False, False, False, True ]
]

ag_equiv = [
    [True, False, False, False, False],
    [False, True, False, False, False],
    [False, False, True, False, False],
    [False, False, False, True, False],
    [False, False, False, False, True]
]

ag_complement = [
    [{'positive': {3}}, {'positive': {9}}, {'negative': {5}}, {'negative': {8}}],
    [{'positive': {9}}, {'negative': {8}}, {'positive': {3, 7}},
     {'positive': {7}, 'negative': {5}}],
    [{'positive': {2, 7}}, {'positive': {3, 7}}],
    [{'positive': {7, 6}, 'negative': {5}},
     {'positive': {7, 6}, 'negative': {8}}, {'positive': {7, 6, 3}},
     {'positive': {7, 6, 9}}],
    [{'positive': {2}, 'negative': {5, 12}},
     {'positive': {2, 7}, 'negative': {5}},
     {'positive': {2, 10}, 'negative': {5}}, {'positive': {2}, 'negative': {8}},
     {'positive': {3}, 'negative': {8}}, {'positive': {3}, 'negative': {12}},
     {'positive': {3, 7}}, {'positive': {3, 9}}, {'positive': {3, 10}},
     {'positive': {2, 9}}]
]

term_box_data = [
    ({'base': [0, 0, -1], 'ex': [1.5, 0, 0], 'ey': [0, 1.5, 0], 'ez': [0, 0, 2]},
     [(-1, [{}]), (1, [{}]), (1, [{}]), (0, [{'negative': {2}}]),
      (-1, [{'negative': {7}}, {'positive': {12}}]),
      (-1, [{'negative': {7}}]), (-1, [{'negative': {7}}]),
      (-1, [{'negative': {6}}])]),
    ({'base': [2.5, 0, -1], 'ex': [3, 0, 0], 'ey': [0, 2.5, 0], 'ez': [0, 0, 4]},
     [(-1, [{}]), (0, [{'negative': {3, 9}}]), (0, [{'negative': {3, 9}}]),
      (-1, [{'negative': {2}}]), (0, [{'negative': {7, 9, 10}}]),
      (0, [{'negative': {7, 9}}]), (0, [{'negative': {7}}]),
      (0, [{'negative': {6}}])])
]

term_intersection_ans = [
    [{}, {}, {}, {}, {}, {}, {}],
    [{}, {'positive': {5, 8, 11}, 'negative': {3, 9, 13}},
     {'positive': {5, 8}, 'negative': {2, 3, 9}},
     {'positive': {5, 8, 12}, 'negative': {3, 7, 9, 10}},
     {'positive': {5, 8}, 'negative': {3, 7, 9}},
     {'positive': {5, 8}, 'negative': {3, 7, 9}},
     {'positive': {5, 8}, 'negative': {3, 6, 9}}],
    [{}, {'positive': {5, 8, 11}, 'negative': {3, 9, 13}},
     {'positive': {5, 8, 11}, 'negative': {2, 3, 9, 13}},
     {'positive': {5, 8, 11, 12}, 'negative': {3, 7, 9, 10, 13}},
     {'positive': {5, 8, 11}, 'negative': {3, 7, 9, 13}},
     {'positive': {5, 8, 11}, 'negative': {3, 7, 9, 13}},
     {'positive': {5, 8, 11}, 'negative': {3, 6, 9, 13}}],
    [{}, {'positive': {5, 8}, 'negative': {2, 3, 9}},
     {'positive': {5, 8, 11}, 'negative': {2, 3, 9, 13}},
     {'positive': {8, 12}, 'negative': {2, 3, 9, 7, 10}},
     {'positive': {8}, 'negative': {2, 3, 7, 9}}, {'negative': {2, 3, 7}},
     {'negative': {2, 3, 6}}],
    [{}, {'positive': {5, 8, 12}, 'negative': {3, 7, 9, 10}},
     {'positive': {5, 8, 11, 12}, 'negative': {3, 7, 9, 10, 13}},
     {'positive': {8, 12}, 'negative': {2, 3, 9, 7, 10}},
     {'positive': {8, 12}, 'negative': {9, 7, 10}},
     {'positive': {8, 12}, 'negative': {9, 7, 10}},
     {'positive': {8, 12}, 'negative': {9, 6, 7, 10}}],
    [{}, {'positive': {5, 8}, 'negative': {3, 7, 9}},
     {'positive': {5, 8, 11}, 'negative': {3, 7, 9, 13}},
     {'positive': {8}, 'negative': {2, 3, 7, 9}},
     {'positive': {12, 8}, 'negative': {7, 9, 10}},
     {'positive': {8}, 'negative': {7, 9}},
     {'positive': {8}, 'negative': {6, 7, 9}}],
    [{}, {'positive': {5, 8}, 'negative': {3, 7, 9}},
     {'positive': {5, 8, 11}, 'negative': {3, 7, 9, 13}},
     {'negative': {2, 3, 7}},
     {'positive': {8, 12}, 'negative': {9, 7, 10}},
     {'positive': {8}, 'negative': {7, 9}},
     {'negative': {6, 7}}],
    [{}, {'positive': {5, 8}, 'negative': {3, 6, 9}},
     {'positive': {5, 8, 11}, 'negative': {3, 6, 9, 13}},
     {'negative': {2, 3, 6}},
     {'positive': {8, 12}, 'negative': {9, 6, 7, 10}},
     {'positive': {8}, 'negative': {6, 7, 9}},
     {'negative': {7, 6}}]
]

term_complexity_ans = [0, 4, 6, 2, 5, 3, 1, 1]
ag_complexity_ans = [4, 7, 3, 6, 11]

is_subset_ans = [
    [True,  True,  True,  True,  True,  True,  True,  True],
    [False, True, False, False, False, False, False, False],
    [False, True,  True, False, False, False, False, False],
    [False, False, False, True, False, False, False, False],
    [False, False, False, False, True,  True,  True, False],
    [False, False, False, False, False, True,  True, False],
    [False, False, False, False, False, False, True, False],
    [False, False, False, False, False, False, False, True]
]

is_empty_ans = [True, False, False, False, False, False, False, False]

term_complement_ans = [
    {},  # Empty set
    {'negative': {5, 8}, 'positive': {3, 9}},
    {'negative': {5, 8, 11}, 'positive': {3, 9, 13}},
    {'positive': {2, 3}},
    {'negative': {12, 8}, 'positive': {9, 7, 10}},
    {'negative': {8}, 'positive': {7, 9}},
    {'positive': {7}}, {'positive': {6}}
]