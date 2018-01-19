import numpy as np

data = {
    'Plane': [
        ('PX', [5.3], {'_v': [1, 0, 0], '_k': -5.3}),
        ('PY', [5.4], {'_v': [0, 1, 0], '_k': -5.4}),
        ('PZ', [5.5], {'_v': [0, 0, 1], '_k': -5.5}),
        ('P', [3.2, -1.4, 5.7, -4.8], {'_v': [3.2, -1.4, 5.7], '_k': 4.8}),
        ('X', [5.6, 6.7], {'_v': [1, 0, 0], '_k': -5.6}),
        ('Y', [5.7, 6.8], {'_v': [0, 1, 0], '_k': -5.7}),
        ('Z', [5.8, -6.9], {'_v': [0, 0, 1], '_k': -5.8}),
        ('X', [5.6, 6.7, 5.6, -7.9], {'_v': [1, 0, 0], '_k': -5.6}),
        ('Y', [5.7, 6.8, 5.7, 6.2], {'_v': [0, 1, 0], '_k': -5.7}),
        ('Z', [5.8, -6.9, 5.8, -9.9], {'_v': [0, 0, 1], '_k': -5.8})
    ],
    'Sphere': [
        ('SO', [6.1], {'_center': [0, 0, 0], '_radius': 6.1}),
        ('SX', [-3.4, 6.2], {'_center': [-3.4, 0, 0], '_radius': 6.2}),
        ('SY', [3.5, 6.3], {'_center': [0, 3.5, 0], '_radius': 6.3}),
        ('SZ', [-3.6, 6.4], {'_center': [0, 0, -3.6], '_radius': 6.4}),
        ('S', [3.7, -3.8, 3.9, 6.5], {'_center': [3.7, -3.8, 3.9], '_radius': 6.5})
    ],
    'Cylinder': [
        ('CX', [6.6], {'_pt': [0, 0, 0], '_axis': [1, 0, 0], '_radius': 6.6}),
        ('CY', [6.7], {'_pt': [0, 0, 0], '_axis': [0, 1, 0], '_radius': 6.7}),
        ('CZ', [6.8], {'_pt': [0, 0, 0], '_axis': [0, 0, 1], '_radius': 6.8}),
        ('C/X', [4.0, -4.1, 6.9], {'_pt': [0, 4.0, -4.1], '_axis': [1, 0, 0], '_radius': 6.9}),
        ('C/Y', [-4.2, 4.3, 7.0], {'_pt': [-4.2, 0, 4.3], '_axis': [0, 1, 0], '_radius': 7.0}),
        ('C/Z', [4.4, 4.5, 7.1], {'_pt': [4.4, 4.5, 0], '_axis': [0, 0, 1], '_radius': 7.1})
    ],
    'Cone': [
        ('KX', [4.6, 0.33], {'_apex': [4.6, 0, 0], '_axis': [1, 0, 0], '_t2': 0.33}),
        ('KY', [4.7, 0.33], {'_apex': [0, 4.7, 0], '_axis': [0, 1, 0], '_t2': 0.33}),
        ('KZ', [-4.8, 0.33], {'_apex': [0, 0, -4.8], '_axis': [0, 0, 1], '_t2': 0.33}),
        ('K/X', [4.9, -5.0, 5.1, 0.33], {'_apex': [4.9, -5.0, 5.1], '_axis': [1, 0, 0], '_t2': 0.33}),
        ('K/Y', [-5.0, -5.1, 5.2, 0.33], {'_apex': [-5.0, -5.1, 5.2], '_axis': [0, 1, 0], '_t2': 0.33}),
        ('K/Z', [5.3, 5.4, 5.5, 0.33], {'_apex': [5.3, 5.4, 5.5], '_axis': [0, 0, 1], '_t2': 0.33})
    ],
    'Torus': [
        ('TX', [1, 2, -3, 5, 0.5, 0.8], {'_center': [1, 2, -3], '_axis': [1, 0, 0], '_R': 5, '_a': 0.5, '_b': 0.8}),
        ('TY', [-4, 5, -6, 3, 0.9, 0.2], {'_center': [-4, 5, -6], '_axis': [0, 1, 0], '_R': 3, '_a': 0.9, '_b': 0.2}),
        ('TZ', [0, -3, 5, 1, 0.1, 0.2], {'_center': [0, -3, 5], '_axis': [0, 0, 1], '_R': 1, '_a': 0.1, '_b': 0.2})
    ],
    'GQuadratic': [
        ('SQ', [0.5, -2.5, 3.0, 1.1, -1.3, -5.4, -7.0, 3.2, -1.7, 8.4],
         {'_m': np.diag([0.5, -2.5, 3.0]), '_v': 2 * np.array([1.1 - 0.5 * 3.2, -1.3 - 2.5 * 1.7, -5.4 - 3.0 * 8.4]), '_k': 0.5 * 3.2 ** 2 - 2.5 * 1.7 ** 2 + 3.0 * 8.4 ** 2 - 7.0 - 2 * (1.1 * 3.2 + 1.3 * 1.7 - 5.4 * 8.4)}),
        ('GQ', [1, 2, 3, 4, 5, 6, 7, 8, 9, -10],
         {'_m': [[1, 2, 3], [2, 2, 2.5], [3, 2.5, 3]], '_v': [7, 8, 9], '_k': -10}),
        ('X', [1, 3.2, -2, 3.2], {'_m': np.diag([0, 1, 1]), '_v': [0, 0, 0], '_k': -3.2 ** 2}),
        ('Y', [-1, 3.3, -2, 3.3], {'_m': np.diag([1, 0, 1]), '_v': [0, 0, 0], '_k': -3.3 ** 2}),
        ('Z', [1, 3.4, 2, 3.4], {'_m': np.diag([1, 1, 0]), '_v': [0, 0, 0], '_k': -3.4 ** 2}),
        ('X', [1, 3, 2, 2], {'_m': np.diag([-1, 1, 1]), '_v': [8, 0, 0], '_k': -16}),
        ('Y', [1, 1, 2, 2], {'_m': np.diag([1, -1, 1]), '_v': [0, 0, 0], '_k': 0}),
        ('Z', [0, 1, 1, 2], {'_m': np.diag([1, 1, -1]), '_v': [0, 0, -2], '_k': -1})
    ]
}