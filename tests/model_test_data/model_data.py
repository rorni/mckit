case_names = ['model1']


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

read_mcnp_ans = {
    'model1': {
        'title': 'model 1',
        'cells': {
            1: {
                'MAT': 1, 'RHO': -1.0,
                'geometry': [1, 2, 'C', 'I', 3, 'I', 4, 'U']
            },
            2: {
                'MAT': 2, 'RHO': -2.0, 'TRCL': 3,
                'geometry': [4, 5, 'I', 6, 'C', 'I', 7, 'U']
            },
            3: {
                'geometry': [1, 2, '#', 'I', 7, '#', 'I'], 'U': 1
            },
            4: {
                'MAT': 3, 'RHO': -4, 'U': 1, 'geometry': [2, 4, 'I', 6, 'I']
            },
            5: {
                'geometry': [1, 3, 'C', 'I', 5, '#', 'I'], 'U': 2
            },
            6: {
                'U': 2, 'TRCL': 4, 'FILL': {'universe': 1},
                'geometry': [2, 4, 'C', 'I', 7, 'I']
            },
            7: {
                'U': 3, 'FILL': {'universe': 4, 'transform': 3},
                'geometry': [2, 4, '#', 'I', 6, 'I']
            },
            8: {
                'U':3, 'MAT': 4, 'RHO': -4, 'geometry': [1, 5, 'I', 7, 'C', 'I']
            },
            9: {
                'MAT': 2, 'RHO': -2, 'U': 4, 'geometry': [2, 5, 'C', 'I', 7, 'I']
            },
            10: {
                'U': 4, 'geometry': [5, 6, 'C', 'I'],
                'FILL': {'universe': 1, 'transform': {'translation': [7, 3, 1]}}
            },
            11: {
                'geometry': [6, 7, 'I'], 'FILL': {'universe': 2}
            },
            12: {
                'geometry': [5, 7, 'I'], 'FILL': {'universe': 3}
            }
        },
        'surfaces': {
            1: ('PX', [-5], {'transform': 1}),
            2: ('PY', [-6], {'transform': 2}),
            3: ('PZ', [-7], {}),
            4: ('PZ', [7], {}),
            5: ('PY', [5], {}),
            6: ('PX', [6], {}),
            7: ('SO', [8], {})
        },
        'data': {
            'TR': {
                1: {'translation': [5, -3, 4]},
                2: {'translation': [1, 1, 1], 'indegrees': True,
                    'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]},
                3: {'translation': [1, 2, 3], 'indegrees': True,
                    'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]},
                4: {'translation': [-1, -2, -3], 'indegrees': True,
                    'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
            },
            'M': {
                1: {'atomic': [(1000, 2), (8000, 1)]},
                2: {'wgt': [(7000, 75.5), (8000, 23.15), (40000, 1.292)]},
                3: {'atomic': [(1000, 2), (6000, 1)]},
                4: {'atomic': [(6012, 1, '50C')]}
            },
            'MODE': ['N']
        }
    }
}

extract_submodel_ans = {
    'model1': {
        (1, True): {
            'title': 'Universe 1',
            'cells': {
                3: {
                    'geometry': [1, 2, '#', 'I', 7, '#', 'I']
                },
                4: {
                    'MAT': 3, 'RHO': -4, 'geometry': [2, 4, 'I', 6, 'I']
                }
            },
            'surfaces': {
                1: ('PX', [-5], {'transform': 1}),
                2: ('PY', [-6], {'transform': 2}),
                4: ('PZ', [7], {}),
                6: ('PX', [6], {}),
            },
            'data': {
                'TR': {
                    1: {'translation': [5, -3, 4]},
                    2: {'translation': [1, 1, 1], 'indegrees': True,
                        'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]},
                },
                'M': {
                    3: {'atomic': [(1000, 2), (6000, 1)]},
                },
                'MODE': ['N']
            }
        },
        (1, False): {
            'title': 'Universe 1',
            'cells': {
                3: {
                    'geometry': [1, 2, '#', 'I', 7, '#', 'I']
                },
                4: {
                    'MAT': 3, 'RHO': -4, 'geometry': [2, 4, 'I', 6, 'I']
                }
            },
            'surfaces': {
                1: ('PX', [-5], {'transform': 1}),
                2: ('PY', [-6], {'transform': 2}),
                4: ('PZ', [7], {}),
                6: ('PX', [6], {})
            },
            'data': {
                'TR': {
                    1: {'translation': [5, -3, 4]},
                    2: {'translation': [1, 1, 1], 'indegrees': True,
                        'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
                },
                'M': {
                    3: {'atomic': [(1000, 2), (6000, 1)]}
                },
                'MODE': ['N']
            }
        },
        (2, True): {
            'title': 'Universe 2',
            'cells': {
                3: {
                    'geometry': [1, 2, '#', 'I', 7, '#', 'I'], 'U': 1
                },
                4: {
                    'MAT': 3, 'RHO': -4, 'U': 1, 'geometry': [2, 4, 'I', 6, 'I']
                },
                5: {
                    'geometry': [1, 3, 'C', 'I', 5, '#', 'I']
                },
                6: {
                    'TRCL': 4, 'FILL': {'universe': 1},
                    'geometry': [2, 4, 'C', 'I', 7, 'I']
                }
            },
            'surfaces': {
                1: ('PX', [-5], {'transform': 1}),
                2: ('PY', [-6], {'transform': 2}),
                3: ('PZ', [-7], {}),
                4: ('PZ', [7], {}),
                6: ('PX', [6], {}),
                7: ('SO', [8], {})
            },
            'data': {
                'TR': {
                    1: {'translation': [5, -3, 4]},
                    2: {'translation': [1, 1, 1], 'indegrees': True,
                        'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]},
                    4: {'translation': [-1, -2, -3], 'indegrees': True,
                        'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
                },
                'M': {
                    3: {'atomic': [(1000, 2), (6000, 1)]}
                },
                'MODE': ['N']
            }
        },
        (2, False): {
            'title': 'Universe 2',
            'cells': {
                5: {
                    'geometry': [1, 3, 'C', 'I', 5, '#', 'I']
                },
                6: {
                    'TRCL': 4, 'geometry': [2, 4, 'C', 'I', 7, 'I']
                }
            },
            'surfaces': {
                1: ('PX', [-5], {'transform': 1}),
                2: ('PY', [-6], {'transform': 2}),
                3: ('PZ', [-7], {}),
                4: ('PZ', [7], {}),
                7: ('SO', [8], {})
            },
            'data': {
                'TR': {
                    1: {'translation': [5, -3, 4]},
                    2: {'translation': [1, 1, 1], 'indegrees': True,
                        'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]},
                    4: {'translation': [-1, -2, -3], 'indegrees': True,
                        'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
                },
                'MODE': ['N']
            }
        },
        (3, True): {
            'title': 'Universe 3',
            'cells': {
                3: {
                    'geometry': [1, 2, '#', 'I', 7, '#', 'I'], 'U': 1
                },
                4: {
                    'MAT': 3, 'RHO': -4, 'U': 1, 'geometry': [2, 4, 'I', 6, 'I']
                },
                7: {
                    'FILL': {'universe': 4, 'transform': 3},
                    'geometry': [2, 4, '#', 'I', 6, 'I']
                },
                8: {
                    'MAT': 4, 'RHO': -4,
                    'geometry': [1, 5, 'I', 7, 'C', 'I']
                },
                9: {
                    'MAT': 2, 'RHO': -2, 'U': 4,
                    'geometry': [2, 5, 'C', 'I', 7, 'I']
                },
                10: {
                    'U': 4, 'geometry': [5, 6, 'C', 'I'],
                    'FILL': {'universe': 1,
                             'transform': {'translation': [7, 3, 1]}}
                }
            },
            'surfaces': {
                1: ('PX', [-5], {'transform': 1}),
                2: ('PY', [-6], {'transform': 2}),
                4: ('PZ', [7], {}),
                5: ('PY', [5], {}),
                6: ('PX', [6], {}),
                7: ('SO', [8], {})
            },
            'data': {
                'TR': {
                    1: {'translation': [5, -3, 4]},
                    2: {'translation': [1, 1, 1], 'indegrees': True,
                        'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]},
                    3: {'translation': [1, 2, 3], 'indegrees': True,
                        'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
                },
                'M': {
                    2: {'wgt': [(7000, 75.5), (8000, 23.15), (40000, 1.292)]},
                    3: {'atomic': [(1000, 2), (6000, 1)]},
                    4: {'atomic': [(6012, 1, '50C')]}
                },
                'MODE': ['N']
            }
        },
        (3, False): {
            'title': 'Universe 3',
            'cells': {
                7: {
                    'geometry': [2, 4, '#', 'I', 6, 'I']
                },
                8: {
                    'MAT': 4, 'RHO': -4,
                    'geometry': [1, 5, 'I', 7, 'C', 'I']
                }
            },
            'surfaces': {
                1: ('PX', [-5], {'transform': 1}),
                2: ('PY', [-6], {'transform': 2}),
                5: ('PY', [5], {}),
                6: ('PX', [6], {}),
                7: ('SO', [8], {})
            },
            'data': {
                'TR': {
                    1: {'translation': [5, -3, 4]},
                    2: {'translation': [1, 1, 1], 'indegrees': True,
                        'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
                },
                'M': {
                    4: {'atomic': [(6012, 1, '50C')]}
                },
                'MODE': ['N']
            }
        },
        (4, True): {
            'title': 'Universe 4',
            'cells': {
                3: {
                    'geometry': [1, 2, '#', 'I', 7, '#', 'I'], 'U': 1
                },
                4: {
                    'MAT': 3, 'RHO': -4, 'U': 1, 'geometry': [2, 4, 'I', 6, 'I']
                },
                9: {
                    'MAT': 2, 'RHO': -2, 'geometry': [2, 5, 'C', 'I', 7, 'I']
                },
                10: {
                    'geometry': [5, 6, 'C', 'I'],
                    'FILL': {'universe': 1,
                             'transform': {'translation': [7, 3, 1]}}
                }
            },
            'surfaces': {
                1: ('PX', [-5], {'transform': 1}),
                2: ('PY', [-6], {'transform': 2}),
                4: ('PZ', [7], {}),
                5: ('PY', [5], {}),
                6: ('PX', [6], {}),
                7: ('SO', [8], {})
            },
            'data': {
                'TR': {
                    1: {'translation': [5, -3, 4]},
                    2: {'translation': [1, 1, 1], 'indegrees': True,
                        'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
                },
                'M': {
                    2: {'wgt': [(7000, 75.5), (8000, 23.15), (40000, 1.292)]},
                    3: {'atomic': [(1000, 2), (6000, 1)]}
                },
                'MODE': ['N']
            }
        },
        (4, False): {
            'title': 'Universe 4',
            'cells': {
                9: {
                    'MAT': 2, 'RHO': -2, 'geometry': [2, 5, 'C', 'I', 7, 'I']
                },
                10: {
                    'geometry': [5, 6, 'C', 'I']
                }
            },
            'surfaces': {
                2: ('PY', [-6], {'transform': 2}),
                5: ('PY', [5], {}),
                6: ('PX', [6], {}),
                7: ('SO', [8], {})
            },
            'data': {
                'TR': {
                    2: {'translation': [1, 1, 1], 'indegrees': True,
                        'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
                },
                'M': {
                    2: {'wgt': [(7000, 75.5), (8000, 23.15), (40000, 1.292)]}
                },
                'MODE': ['N']
            }
        }
    }
}

model_list_universes_ans = {
    'model1': [1, 2, 3, 4]
}


