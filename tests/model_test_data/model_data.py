get_surface_indices_ans = {
    'model1': {
        1: {1, 2, 3, 4},
        2: {4, 5, 6, 7},
        3: {1},
        4: {2, 4, 6},
        5: {1, 3},
        6: {2, 4, 7},
        7: {2, 6},
        8: {1, 5, 7},
        9: {2, 5, 7},
        10: {5, 6},
        11: {6, 7},
        12: {5, 7}
    }
}

get_universe_list_ans = {
    'model1': [0, 1, 2, 3, 4]
}

contained_universes_ans = {
    'model1': {
        0: {1, 2, 3, 4},
        1: set(),
        2: {1},
        3: {4, 1},
        4: {1}
    }
}

get_universe_model_ans = {
    'model1': {
        1: {
            'title': 'Universe 1',
            'cells': {3, 4},
            'surfaces': {1, 2, 4, 6},
            'transform': {1, 2},
            'material': {3}
        },
        2: {
            'title': 'Universe 2',
            'cells': {3, 4, 5, 6},
            'surfaces': {1, 2, 3, 4, 6, 7},
            'transform': {1, 2, 4},
            'material': {3}
        },
        3: {
            'title': 'Universe 3',
            'cells': {3, 4, 7, 8, 9, 10},
            'surfaces': {1, 2, 4, 5, 6, 7},
            'transform': {1, 2, 3, 4},
            'material': {2, 3, 4}
        },
        4: {
            'title': 'Universe 4',
            'cells': {3, 4, 9, 10},
            'surfaces': {1, 2, 4, 5, 7, 6},
            'transform': {1, 2, 4},
            'material': {2, 3}
        }
    }
}