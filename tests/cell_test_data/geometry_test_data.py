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
    13: ('py', [3.5])
}

terms = [
    {},  # Empty set
    {'positive': {5, 8}, 'negative': {3, 9}},
    {'positive': {5, 8, 11}, 'negative': {3, 9, 13}},
    {'negative': {2}},
    {'positive': {12, 8}, 'negative': {9, 7, 10}},
    {'positive': {8}, 'negative': {7, 9}},
    {'negative': {7}}
]

box_data = [
    ({'base': [0, 0, -1], 'ex': [1.5, 0, 0], 'ey': [0, 1.5, 0], 'ez': [0, 0, 2]},
     [(-1, [{}]), (1, [{}]), (1, [{}]), (0, [{'negative': {2}}]),
      (-1, [{'negative': {7}}, {'positive': {12}}]),
      (-1, [{'negative': {7}}]), (-1, [{'negative': {7}}])]),
    ({'base': [2.5, 0, -1], 'ex': [3, 0, 0], 'ey': [0, 2.5, 0], 'ez': [0, 0, 4]},
     [(-1, [{}]), (0, [{'negative': {3, 9}}]), (0, [{'negative': {3, 9}}]),
      (-1, [{'negative': {2}}]), (0, [{'negative': {7, 9, 10}}]),
      (0, [{'negative': {7, 9}}]), (0, [{'negative': {7}}])])
]

intersection_ans = [
    [{}, {}, {}, {}, {}, {}],
    [{}, {'positive': {5, 8, 11}, 'negative': {3, 9, 13}},
     {'positive': {5, 8}, 'negative': {2, 3, 9}},
     {'positive': {5, 8, 12}, 'negative': {3, 7, 9, 10}},
     {'positive': {5, 8}, 'negative': {3, 7, 9}},
     {'positive': {5, 8}, 'negative': {3, 7, 9}}],
    [{}, {'positive': {5, 8, 11}, 'negative': {3, 9, 13}},
     {'positive': {5, 8, 11}, 'negative': {2, 3, 9, 13}},
     {'positive': {5, 8, 11, 12}, 'negative': {3, 7, 9, 10, 13}},
     {'positive': {5, 8, 11}, 'negative': {3, 7, 9, 13}},
     {'positive': {5, 8, 11}, 'negative': {3, 7, 9, 13}}],
    [{}, {'positive': {5, 8}, 'negative': {2, 3, 9}},
     {'positive': {5, 8, 11}, 'negative': {2, 3, 9, 13}},
     {'positive': {8, 12}, 'negative': {2, 9, 7, 10}},
     {'positive': {8}, 'negative': {2, 7, 9}}, {'negative': {2, 7}}],
    [{}, {'positive': {5, 8, 12}, 'negative': {3, 7, 9, 10}},
     {'positive': {5, 8, 11, 12}, 'negative': {3, 7, 9, 10, 13}},
     {'positive': {8, 12}, 'negative': {2, 9, 7, 10}},
     {'positive': {8, 12}, 'negative': {9, 7, 10}},
     {'positive': {8, 12}, 'negative': {9, 7, 10}}],
    [{}, {'positive': {5, 8}, 'negative': {3, 7, 9}},
     {'positive': {5, 8, 11}, 'negative': {3, 7, 9, 13}},
     {'positive': {8}, 'negative': {2, 7, 9}},
     {'positive': {12, 8}, 'negative': {7, 9, 10}},
     {'positive': {8}, 'negative': {7, 9}}],
    [{}, {'positive': {5, 8}, 'negative': {3, 7, 9}},
     {'positive': {5, 8, 11}, 'negative': {3, 7, 9, 13}},
     {'negative': {2, 7}},
     {'positive': {8, 12}, 'negative': {9, 7, 10}},
     {'positive': {8}, 'negative': {7, 9}}]
]

complexity_ans = [0, 4, 6, 1, 5, 3, 1]

is_subset_ans = [
    [True,  True,  True,  True,  True,  True,  True],
    [False, True, False, False, False, False, False],
    [False, True,  True, False, False, False, False],
    [False, False, False, True, False, False, False],
    [False, False, False, False, True,  True,  True],
    [False, False, False, False, False, True,  True],
    [False, False, False, False, False, False, True]
]

is_empty_ans = [True, False, False, False, False, False, False]

complement_ans = [
    {},  # Empty set
    {'negative': {5, 8}, 'positive': {3, 9}},
    {'negative': {5, 8, 11}, 'positive': {3, 9, 13}},
    {'positive': {2}},
    {'negative': {12, 8}, 'positive': {9, 7, 10}},
    {'negative': {8}, 'positive': {7, 9}},
    {'positive': {7}}
]