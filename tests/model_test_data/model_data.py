case_names = ['model1']

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

get_universe_dependencies_ans = {
    'model1': {
        0: {2, 3},
        1: set(),
        2: {1},
        3: {4},
        4: {1}
    }
}

get_contained_cells_ans = {
    'model1': {
        (0, True): {
            0: {1, 2, 11, 12}, 1: {3, 4}, 2: {5, 6}, 3: {7, 8}, 4: {9, 10}
        },
        (0, False): {
            0: {1, 2, 11, 12}
        },
        (1, True): {
            0: {3, 4}
        },
        (1, False): {
            0: {3, 4}
        },
        (2, True): {
            0: {5, 6}, 1: {3, 4}
        },
        (2, False): {
            0: {5, 6}
        },
        (3, True): {
            0: {7, 8}, 4: {9, 10}, 1: {3, 4}
        },
        (3, False): {
            0: {7, 8}
        },
        (4, True): {
            0: {9, 10}, 1: {3, 4}
        },
        (4, False): {
            0: {9, 10}
        }
    }
}

get_contained_compositions_ans = {
    'model1': {
        (0, True): {1, 2, 3, 4},
        (0, False): {1, 2},
        (1, True): {3},
        (1, False): {3},
        (2, True): {3},
        (2, False): set(),
        (3, True): {4, 3, 2},
        (3, False): {4},
        (4, True): {2, 3},
        (4, False): {2}
    }
}

get_contained_transformations_ans = {
    'model1': {
        (0, True): {1, 2, 3, 4},
        (0, False): {1, 2, 3},
        (1, True): {1, 2},
        (1, False): {1, 2},
        (2, True): {1, 2, 4},
        (2, False): {1, 2, 4},
        (3, True): {1, 2, 3},
        (3, False): {1, 2},
        (4, True): {2, 1},
        (4, False): {2}
    }
}

get_contained_surfaces_ans = {
    'model1': {
        (0, True): {1, 2, 3, 4, 5, 6, 7},
        (0, False): {1, 2, 3, 4, 5, 6, 7},
        (1, True): {1, 2, 4, 6},
        (1, False): {1, 2, 4, 6},
        (2, True): {1, 2, 4, 6, 3, 7},
        (2, False): {1, 2, 3, 4, 7},
        (3, True): {1, 2, 4, 6, 5, 7},
        (3, False): {1, 2, 5, 6, 7},
        (4, True): {1, 2, 4, 6, 5, 7},
        (4, False): {2, 5, 6, 7}
    }
}
