# -*- coding: utf-8 -*-

import unittest

import numpy as np

from mckit.surface import Plane, GQuadratic, Torus, Sphere, Cylinder, Cone, \
    create_surface
from mckit.transformation import Transformation
from tests.surface_test_data import surface_create_data
from tests.surface_test_data import surface_test_point_data
from tests.surface_test_data import surface_test_region_data
from tests.surface_test_data import surfobj_create_data


trs = [
    None,
    Transformation(translation=[1, 2, -3], indegrees=True,
                   rotation=[30, 60, 90, 120, 30, 90, 90, 90, 0])
]

class_apply = {'Plane': Plane, 'Sphere': Sphere, 'Cylinder': Cylinder,
               'Cone': Cone, 'Torus': Torus, 'GQuadratic': GQuadratic}
type_tr = {
    '_center': 'apply2point',
    '_axis': 'apply2vector',
    '_m': 'apply2gq',
    '_pt': 'apply2point',
    '_apex': 'apply2point'
}


class TestSurfaceMethods(unittest.TestCase):
    def test_init_method(self):
        for class_name, test_cases in surfobj_create_data.data.items():
            TClass = class_apply[class_name]
            for i, (*params, answer) in enumerate(test_cases):
                for j, tr in enumerate(trs):
                    msg = class_name + ' case {0}, tr={1}'.format(i, j)
                    with self.subTest(msg=msg):
                        options = {}
                        if tr is not None:
                            options['transform'] = tr
                            if class_name == 'GQuadratic':
                                m, v, k = tr.apply2gq(answer['_m'],
                                                      answer['_v'],
                                                      answer['_k'])
                                answer1 = {'_m': m, '_v': v, '_k': k}
                            else:
                                answer1 = {k: getattr(tr, type_tr[k])(v) for
                                           k, v in answer.items() if
                                           k in type_tr.keys()}
                        else:
                            answer1 = answer
                        surf = TClass(*params, **options)
                        for name, ans_value in answer1.items():
                            sur_value = getattr(surf, name)
                            np.testing.assert_almost_equal(sur_value, ans_value)

    def test_point_test(self):
        for class_name, test_cases in surface_test_point_data.data.items():
            TClass = class_apply[class_name]
            for surf_params, test_points in test_cases:
                surf = TClass(*surf_params)
                for i, (p, ans) in enumerate(test_points):
                    msg = class_name + ' params: {0}, case {1}'.format(surf_params, i)
                    with self.subTest(msg=msg):
                        sense = surf.test_point(p)
                        if isinstance(sense, np.ndarray):
                            for s, a in zip(sense, ans):
                                self.assertEqual(s, a)
                        else:
                            self.assertEqual(sense, ans)

    def test_transform(self):
        for class_name, test_cases in surfobj_create_data.data.items():
            TClass = class_apply[class_name]
            for i, (*params, answer) in enumerate(test_cases):
                surf = TClass(*params)
                for j, tr in enumerate(trs[1:]):
                    msg = class_name + ' case {0}, tr={1}'.format(i, j)
                    with self.subTest(msg=msg):
                        surf_tr = surf.transform(tr)
                        if class_name == 'GQuadratic':
                            m, v, k = tr.apply2gq(answer['_m'], answer['_v'], answer['_k'])
                            answer1 = {'_m': m, '_v': v, '_k': k}
                        else:
                            answer1 = {k: getattr(tr, type_tr[k])(v) for k, v in answer.items() if k in type_tr.keys()}
                        for name, ans_value in answer1.items():
                            sur_value = getattr(surf_tr, name)
                            np.testing.assert_almost_equal(sur_value, ans_value)

    def test_region_test(self):
        for class_name, test_cases in surface_test_region_data.data.items():
            TClass = class_apply[class_name]
            for i, (*params, ans) in enumerate(test_cases):
                msg = class_name + ' case {0}'.format(i)
                with self.subTest(msg=msg):
                    surf = TClass(*params)
                    result = surf.test_region(surface_test_region_data.region)
                    self.assertEqual(result, ans)


class TestSurfaceCreation(unittest.TestCase):
    def test_surface_creation(self):
        for class_name, test_cases in surface_create_data.data.items():
            for i, (kind, params, answer) in enumerate(test_cases):
                msg = class_name + ' case {0}'.format(i)
                with self.subTest(msg=msg):
                    surf = create_surface(kind, *params)
                    for name, ans_value in answer.items():
                        sur_value = getattr(surf, name)
                        np.testing.assert_almost_equal(sur_value, ans_value)


if __name__ == '__main__':
    unittest.main()
