import pytest
import numpy as np
import pickle

from mckit.transformation import Transformation
from mckit.surface import Plane, GQuadratic, Torus, Sphere, Cylinder, Cone, \
    create_surface
from mckit.geometry import Box


@pytest.fixture(scope='module', params=[
    {},
    {'translation': [1, 2, -3], 'indegrees': True,
     'rotation': [30, 60, 90, 120, 30, 90, 90, 90, 0]}
])
def transform(request):
    return Transformation(**request.param)


@pytest.fixture(scope='module')
def box():
    return Box([0, 0, 0], 2, 2, 2)


@pytest.mark.parametrize('cls, kind, params, expected', [
      (Plane, 'PX', [5.3], {'_v': [1, 0, 0], '_k': -5.3}),
      (Plane, 'PY', [5.4], {'_v': [0, 1, 0], '_k': -5.4}),
      (Plane, 'PZ', [5.5], {'_v': [0, 0, 1], '_k': -5.5}),
      (Plane, 'P', [3.2, -1.4, 5.7, -4.8], {'_v': [3.2, -1.4, 5.7], '_k': 4.8}),
      (Plane, 'X', [5.6, 6.7], {'_v': [1, 0, 0], '_k': -5.6}),
      (Plane, 'Y', [5.7, 6.8], {'_v': [0, 1, 0], '_k': -5.7}),
      (Plane, 'Z', [5.8, -6.9], {'_v': [0, 0, 1], '_k': -5.8}),
      (Plane, 'X', [5.6, 6.7, 5.6, -7.9], {'_v': [1, 0, 0], '_k': -5.6}),
      (Plane, 'Y', [5.7, 6.8, 5.7, 6.2], {'_v': [0, 1, 0], '_k': -5.7}),
      (Plane, 'Z', [5.8, -6.9, 5.8, -9.9], {'_v': [0, 0, 1], '_k': -5.8}),
      (Sphere, 'SO', [6.1], {'_center': [0, 0, 0], '_radius': 6.1}),
      (Sphere, 'SX', [-3.4, 6.2], {'_center': [-3.4, 0, 0], '_radius': 6.2}),
      (Sphere, 'SY', [3.5, 6.3], {'_center': [0, 3.5, 0], '_radius': 6.3}),
      (Sphere, 'SZ', [-3.6, 6.4], {'_center': [0, 0, -3.6], '_radius': 6.4}),
      (Sphere, 'S', [3.7, -3.8, 3.9, 6.5], {'_center': [3.7, -3.8, 3.9], '_radius': 6.5}),
      (Cylinder, 'CX', [6.6], {'_pt': [0, 0, 0], '_axis': [1, 0, 0], '_radius': 6.6}),
      (Cylinder, 'CY', [6.7], {'_pt': [0, 0, 0], '_axis': [0, 1, 0], '_radius': 6.7}),
      (Cylinder, 'CZ', [6.8], {'_pt': [0, 0, 0], '_axis': [0, 0, 1], '_radius': 6.8}),
      (Cylinder, 'C/X', [4.0, -4.1, 6.9], {'_pt': [0, 4.0, -4.1], '_axis': [1, 0, 0], '_radius': 6.9}),
      (Cylinder, 'C/Y', [-4.2, 4.3, 7.0], {'_pt': [-4.2, 0, 4.3], '_axis': [0, 1, 0], '_radius': 7.0}),
      (Cylinder, 'C/Z', [4.4, 4.5, 7.1], {'_pt': [4.4, 4.5, 0], '_axis': [0, 0, 1], '_radius': 7.1}),
      (Cylinder, 'X', [1.2, 3.4, 8.4, 3.4], {'_pt': [0, 0, 0], '_axis': [1, 0, 0], '_radius': 3.4}),
      (Cylinder, 'Y', [1.2, 3.4, 8.4, 3.4], {'_pt': [0, 0, 0], '_axis': [0, 1, 0], '_radius': 3.4}),
      (Cylinder, 'Z', [1.2, 3.4, 8.4, 3.4], {'_pt': [0, 0, 0], '_axis': [0, 0, 1], '_radius': 3.4}),
      (Cone, 'KX', [4.6, 0.33], {'_apex': [4.6, 0, 0], '_axis': [1, 0, 0], '_t2': 0.33, '_sheet': 0}),
      (Cone, 'KY', [4.7, 0.33], {'_apex': [0, 4.7, 0], '_axis': [0, 1, 0], '_t2': 0.33, '_sheet': 0}),
      (Cone, 'KZ', [-4.8, 0.33], {'_apex': [0, 0, -4.8], '_axis': [0, 0, 1], '_t2': 0.33, '_sheet': 0}),
      (Cone, 'K/X', [4.9, -5.0, 5.1, 0.33], {'_apex': [4.9, -5.0, 5.1], '_axis': [1, 0, 0], '_t2': 0.33, '_sheet': 0}),
      (Cone, 'K/Y', [-5.0, -5.1, 5.2, 0.33], {'_apex': [-5.0, -5.1, 5.2], '_axis': [0, 1, 0], '_t2': 0.33, '_sheet': 0}),
      (Cone, 'K/Z', [5.3, 5.4, 5.5, 0.33], {'_apex': [5.3, 5.4, 5.5], '_axis': [0, 0, 1], '_t2': 0.33, '_sheet': 0}),
      (Cone, 'KX', [4.6, 0.33, +1], {'_apex': [4.6, 0, 0], '_axis': [1, 0, 0], '_t2': 0.33, '_sheet': +1}),
      (Cone, 'KY', [4.7, 0.33, +1], {'_apex': [0, 4.7, 0], '_axis': [0, 1, 0], '_t2': 0.33, '_sheet': +1}),
      (Cone, 'KZ', [-4.8, 0.33, +1], {'_apex': [0, 0, -4.8], '_axis': [0, 0, 1], '_t2': 0.33, '_sheet': +1}),
      (Cone, 'X', [-1.0, 1.0, 1.0, 2.0], {'_apex': [-3.0, 0, 0], '_axis': [1, 0, 0], '_t2': 0.25, '_sheet': +1}),
      (Cone, 'X', [-2.5, 4.5, -0.5, 3.5], {'_apex': [6.5, 0, 0], '_axis': [1, 0, 0], '_t2': 0.25, '_sheet': -1}),
      (Cone, 'X', [1.0, 2.0, -1.0, 1.0], {'_apex': [-3.0, 0, 0], '_axis': [1, 0, 0], '_t2': 0.25, '_sheet': +1}),
      (Cone, 'X', [-0.5, 3.5, -2.5, 4.5], {'_apex': [6.5, 0, 0], '_axis': [1, 0, 0], '_t2': 0.25, '_sheet': -1}),
      (Cone, 'Y', [-1.0, 1.0, 1.0, 2.0], {'_apex': [0, -3.0, 0], '_axis': [0, 1, 0], '_t2': 0.25, '_sheet': +1}),
      (Cone, 'Y', [-2.5, 4.5, -0.5, 3.5], {'_apex': [0, 6.5, 0], '_axis': [0, 1, 0], '_t2': 0.25, '_sheet': -1}),
      (Cone, 'Y', [1.0, 2.0, -1.0, 1.0], {'_apex': [0, -3.0, 0], '_axis': [0, 1, 0], '_t2': 0.25, '_sheet': +1}),
      (Cone, 'Y', [-0.5, 3.5, -2.5, 4.5], {'_apex': [0, 6.5, 0], '_axis': [0, 1, 0], '_t2': 0.25, '_sheet': -1}),
      (Cone, 'Z', [-1.0, 1.0, 1.0, 2.0], {'_apex': [0, 0, -3.0], '_axis': [0, 0, 1], '_t2': 0.25, '_sheet': +1}),
      (Cone, 'Z', [-2.5, 4.5, -0.5, 3.5], {'_apex': [0, 0, 6.5], '_axis': [0, 0, 1], '_t2': 0.25, '_sheet': -1}),
      (Cone, 'Z', [1.0, 2.0, -1.0, 1.0], {'_apex': [0, 0, -3.0], '_axis': [0, 0, 1], '_t2': 0.25, '_sheet': +1}),
      (Cone, 'Z', [-0.5, 3.5, -2.5, 4.5], {'_apex': [0, 0, 6.5], '_axis': [0, 0, 1], '_t2': 0.25, '_sheet': -1}),
      (Cone, 'K/X', [4.9, -5.0, 5.1, 0.33, +1], {'_apex': [4.9, -5.0, 5.1], '_axis': [1, 0, 0], '_t2': 0.33, '_sheet': +1}),
      (Cone, 'K/Y', [-5.0, -5.1, 5.2, 0.33, +1], {'_apex': [-5.0, -5.1, 5.2], '_axis': [0, 1, 0], '_t2': 0.33, '_sheet': +1}),
      (Cone, 'K/Z', [5.3, 5.4, 5.5, 0.33, +1], {'_apex': [5.3, 5.4, 5.5], '_axis': [0, 0, 1], '_t2': 0.33, '_sheet': +1}),
      (Cone, 'KX', [4.6, 0.33, -1], {'_apex': [4.6, 0, 0], '_axis': [1, 0, 0], '_t2': 0.33, '_sheet': -1}),
      (Cone, 'KY', [4.7, 0.33, -1], {'_apex': [0, 4.7, 0], '_axis': [0, 1, 0], '_t2': 0.33, '_sheet': -1}),
      (Cone, 'KZ', [-4.8, 0.33, -1], {'_apex': [0, 0, -4.8], '_axis': [0, 0, 1], '_t2': 0.33, '_sheet': -1}),
      (Cone, 'K/X', [4.9, -5.0, 5.1, 0.33, -1], {'_apex': [4.9, -5.0, 5.1], '_axis': [1, 0, 0], '_t2': 0.33, '_sheet': -1}),
      (Cone, 'K/Y', [-5.0, -5.1, 5.2, 0.33, -1], {'_apex': [-5.0, -5.1, 5.2], '_axis': [0, 1, 0], '_t2': 0.33, '_sheet': -1}),
      (Cone, 'K/Z', [5.3, 5.4, 5.5, 0.33, -1], {'_apex': [5.3, 5.4, 5.5], '_axis': [0, 0, 1], '_t2': 0.33, '_sheet': -1}),
      (Torus, 'TX', [1, 2, -3, 5, 0.5, 0.8], {'_center': [1, 2, -3], '_axis': [1, 0, 0], '_R': 5, '_a': 0.5, '_b': 0.8}),
      (Torus, 'TY', [-4, 5, -6, 3, 0.9, 0.2], {'_center': [-4, 5, -6], '_axis': [0, 1, 0], '_R': 3, '_a': 0.9, '_b': 0.2}),
      (Torus, 'TZ', [0, -3, 5, 1, 0.1, 0.2], {'_center': [0, -3, 5], '_axis': [0, 0, 1], '_R': 1, '_a': 0.1, '_b': 0.2}),
      (GQuadratic, 'SQ', [0.5, -2.5, 3.0, 1.1, -1.3, -5.4, -7.0, 3.2, -1.7, 8.4], {'_m': np.diag([0.5, -2.5, 3.0]), '_v': 2 * np.array([1.1 - 0.5 * 3.2, -1.3 - 2.5 * 1.7, -5.4 - 3.0 * 8.4]), '_k': 0.5 * 3.2 ** 2 - 2.5 * 1.7 ** 2 + 3.0 * 8.4 ** 2 - 7.0 - 2 * ( 1.1 * 3.2 + 1.3 * 1.7 - 5.4 * 8.4)}),
      (GQuadratic, 'GQ', [1, 2, 3, 4, 5, 6, 7, 8, 9, -10], {'_m': [[1, 2, 3], [2, 2, 2.5], [3, 2.5, 3]], '_v': [7, 8, 9], '_k': -10}),
])
def test_surface_creation(cls, kind, params, expected):
    surf = create_surface(kind, *params)
    assert isinstance(surf, cls)
    for attr_name, attr_value in expected.items():
        surf_attr = getattr(surf, attr_name)
        np.testing.assert_array_almost_equal(surf_attr, attr_value)


class TestPlane:
    @pytest.mark.parametrize('norm, offset, v, k', [
        ([0, 0, 1], -2, np.array([0, 0, 1]), -2),
        ([1, 0, 0], -2, np.array([1, 0, 0]), -2),
        ([0, 1, 0], -2, np.array([0, 1, 0]), -2),
    ])
    def test_init(self, transform, norm, offset, v, k):
        surf = Plane(norm, offset, transform=transform)
        v, k = transform.apply2plane(v, k)
        np.testing.assert_array_almost_equal(v, surf._v)
        np.testing.assert_array_almost_equal(k, surf._k)

    @pytest.mark.parametrize('point, expected', [
        ([1, 0, 0], [+1]), ([-1, 0, 0], [-1]), ([1, 0, 0], [+1]),
        ([-1, 0, 0], [-1]), ([0.1, 0, 0], [+1]), ([-0.1, 0, 0], [-1]),
        ([1.e-6, 100, -300], [+1]), ([-1.e-6, 200, -500], [-1]),
        (np.array([[1, 0, 0], [-1, 0, 0], [0.1, 0, 0], [-0.1, 0, 0],
                   [1.e-6, 100, -300],
                   [-1.e-6, 200, -500]]), np.array([1, -1, 1, -1, 1, -1]))
    ])
    def test_point_test(self, point, expected):
        surf = Plane([1, 0, 0], 0)
        result = surf.test_points(point)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize('norm, offset, ans', [
        ([1, 0, 0], 0, 0),
        ([1, 0, 0], 1.001, 1),
        ([1, 0, 0], -1.001, -1),
        ([1, 0, 0], 0.999, 0),
        ([1, 0, 0], -0.999, 0),
        ([0, 1, 0], 0, 0),
        ([0, 1, 0], 1.001, 1),
        ([0, 1, 0], -1.001, -1),
        ([0, 1, 0], 0.999, 0),
        ([0, 1, 0], -0.999, 0),
        ([0, 0, 1], 0, 0),
        ([0, 0, 1], 1.001, 1),
        ([0, 0, 1], -1.001, -1),
        ([0, 0, 1], 0.999, 0),
        ([0, 0, 1], -0.999, 0),
        ([1, 1, 1], -2.999, 0),
        ([1, 1, 1], -3.001, -1),
        ([1, 1, 1], 2.999, 0),
        ([1, 1, 1], 3.001, 1)
    ])
    def test_box_test(self, box, norm, offset, ans):
        surf = Plane(norm, offset)
        assert surf.test_box(box) == ans

    @pytest.mark.parametrize('norm, offset', [
        ([0, 0, 1], -2),
        ([1, 0, 0], -2),
        ([0, 1, 0], -2),
    ])
    def test_transform(self, transform, norm, offset):
        ans_surf = Plane(norm, offset, transform=transform)
        surf = Plane(norm, offset)
        surf_tr = surf.transform(transform)
        np.testing.assert_array_almost_equal(ans_surf._v, surf_tr._v)
        np.testing.assert_almost_equal(ans_surf._k, surf_tr._k)

    @pytest.mark.parametrize('norm, offset, options', [
        ([0, 0, 1], -2, {}),
        ([1, 0, 0], -2, {'name': 3}),
        ([0, 1, 0], -2, {'name': 4, 'comments': ['abc', 'def']}),
    ])
    def test_pickle(self, norm, offset, options):
        surf = Plane(norm, offset, **options)
        with open('test.pic', 'bw') as f:
            pickle.dump(surf, f, pickle.HIGHEST_PROTOCOL)
        with open('test.pic', 'br') as f:
            surf_un = pickle.load(f)
        np.testing.assert_array_almost_equal(surf._v, surf_un._v)
        np.testing.assert_almost_equal(surf._k, surf_un._k)
        assert surf.options == surf_un.options


class TestSphere:
    @pytest.mark.parametrize('center, radius, c, r', [
        ([1, 2, 3], 5, np.array([1, 2, 3]), 5)
    ])
    def test_init(self, transform, center, radius, c, r):
        surf = Sphere(center, radius, transform=transform)
        c = transform.apply2point(c)
        np.testing.assert_array_almost_equal(c, surf._center)
        np.testing.assert_array_almost_equal(r, surf._radius)

    @pytest.mark.parametrize('point, expected', [
        (np.array([1, 2, 3]), [-1]), (np.array([5.999, 2, 3]), [-1]),
        (np.array([6.001, 2, 3]), [+1]), (np.array([1, 6.999, 3]), [-1]),
        (np.array([1, 7.001, 3]), [+1]), (np.array([1, 2, 7.999]), [-1]),
        (np.array([1, 2, 8.001]), [+1]), (np.array([-3.999, 2, 3]), [-1]),
        (np.array([-4.001, 2, 3]), [+1]), (np.array([1, 2.999, 3]), [-1]),
        (np.array([1, -3.001, 3]), [+1]), (np.array([1, 2, -1.999]), [-1]),
        (np.array([1, 2, -2.001]), [+1]),
        (np.array([[1, 2, 3], [5.999, 2, 3], [6.001, 2, 3], [1, 6.999, 3],
                   [1, 7.001, 3], [1, 2, 7.999], [1, 2, 8.001],
                   [-3.999, 2, 3],
                   [-4.001, 2, 3], [1, 2.999, 3], [1, -3.001, 3],
                   [1, 2, -1.999],
                   [1, 2, -2.001]]),
         np.array([-1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1]))
    ])
    def test_point_test(self, point, expected):
        surf = Sphere([1, 2, 3], 5)
        result = surf.test_points(point)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize('center, radius, ans', [
        (np.array([0, 0, 0]), 1.8, -1), (np.array([0, 0, 0]), 1.7, 0),
        (np.array([0, 0, -2]), 0.999, +1), (np.array([0, 0, -2]), 1.001, 0),
        (np.array([-2, -2, -2]), 1.7, +1), (np.array([-2, -2, -2]), 1.8, 0),
        (np.array([-2, -2, -2]), 5.1, 0), (np.array([-2, -2, -2]), 5.2, -1),
        (np.array([-2, 0, -2]), 1.4, +1), (np.array([-2, 0, -2]), 1.5, 0),
        (np.array([-2, 0, -2]), 4.3, 0), (np.array([-2, 0, -2]), 4.4, -1)
    ])
    def test_box_test(self, box, center, radius, ans):
        surf = Sphere(center, radius)
        assert surf.test_box(box) == ans

    @pytest.mark.parametrize('center, radius', [
        ([1, 2, 3], 5),
    ])
    def test_transform(self, transform, center, radius):
        ans_surf = Sphere(center, radius, transform=transform)
        surf = Sphere(center, radius)
        surf_tr = surf.transform(transform)
        np.testing.assert_array_almost_equal(ans_surf._center, surf_tr._center)
        np.testing.assert_almost_equal(ans_surf._radius, surf_tr._radius)

    @pytest.mark.parametrize('center, radius, options', [
        ([1, 2, 3], 5, {}),
        ([1, 2, 3], 5, {'name': 2}),
        ([1, 2, 3], 5, {'name': 3, 'comment': ['abc', 'def']}),
    ])
    def test_pickle(self, center, radius, options):
        surf = Sphere(center, radius, **options)
        with open('test.pic', 'bw') as f:
            pickle.dump(surf, f, pickle.HIGHEST_PROTOCOL)
        with open('test.pic', 'br') as f:
            surf_un = pickle.load(f)
        np.testing.assert_array_almost_equal(surf._center, surf_un._center)
        np.testing.assert_almost_equal(surf._radius, surf_un._radius)
        assert surf.options == surf_un.options


class TestCylinder:
    @pytest.mark.parametrize('point, axis, radius, pt, ax, rad', [
        ([1, 2, 3], [1, 2, 3], 5, [1, 2, 3], np.array([1, 2, 3]) / np.sqrt(14),
          5)
    ])
    def test_init(self, transform, point, axis, radius, pt, ax, rad):
        surf = Cylinder(point, axis, radius, transform=transform)
        pt = transform.apply2point(pt)
        ax = transform.apply2vector(ax)
        np.testing.assert_array_almost_equal(pt, surf._pt)
        np.testing.assert_array_almost_equal(ax, surf._axis)
        np.testing.assert_array_almost_equal(rad, surf._radius)

    @pytest.mark.parametrize('point, expected', [
        ([0, 3, 4], [-1]), ([-2, 3, 2.001], [-1]), ([-3, 3, 1.999], [+1]),
        ([2, 3, 5.999], [-1]), ([3, 3, 6.001], [+1]), ([4, 1.001, 4], [-1]),
        ([-4, 0.999, 4], [+1]), ([-5, 4.999, 4], [-1]), ([5, 5.001, 4], [+1]),
        (np.array([[0, 3, 4], [-2, 3, 2.001], [-3, 3, 1.999], [2, 3, 5.999],
                   [3, 3, 6.001], [4, 1.001, 4], [-4, 0.999, 4],
                   [-5, 4.999, 4],
                   [5, 5.001, 4]]), [-1, -1, +1, -1, +1, -1, +1, -1, +1])
    ])
    def test_point_test(self, point, expected):
        surf = Cylinder([-1, 3, 4], [1, 0, 0], 2)
        result = surf.test_points(point)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize('point, axis, radius, ans', [
        ([0, 0, 0], [1, 0, 0], 0.5, 0), ([0, 0, 0], [1, 0, 0], 1.4, 0),
        ([0, 0, 0], [1, 0, 0], 1.5, -1), ([0, 1, 1], [1, 0, 0], 0.001, 0),
        ([0, 1, 1], [1, 0, 0], 2.8, 0), ([0, 1, 1], [1, 0, 0], 2.9, -1),
        ([0, 2, 0], [1, 0, 0], 0.999, +1), ([0, 2, 0], [1, 0, 0], 1.001, 0),
        ([0, 2, 0], [1, 0, 0], 3.1, 0), ([0, 2, 0], [1, 0, 0], 3.2, -1),
        ([0, 0, 0], [0, 1, 0], 0.5, 0), ([0, 0, 0], [0, 1, 0], 1.4, 0),
        ([0, 0, 0], [0, 1, 0], 1.5, -1), ([1, 0, 1], [0, 1, 0], 0.001, 0),
        ([1, 0, 1], [0, 1, 0], 2.8, 0), ([1, 0, 1], [0, 1, 0], 2.9, -1),
        ([2, 0, 0], [0, 1, 0], 0.999, +1), ([2, 0, 0], [0, 1, 0], 1.001, 0),
        ([2, 0, 0], [0, 1, 0], 3.1, 0), ([2, 0, 0], [0, 1, 0], 3.2, -1)
    ])
    def test_box_test(self, box, point, axis, radius, ans):
        surf = Cylinder(point, axis, radius)
        assert surf.test_box(box) == ans

    @pytest.mark.parametrize('point, axis, radius', [
        ([1, 2, 3], np.array([1, 2, 3]) / np.sqrt(14), 5),
    ])
    def test_transform(self, transform, point, axis, radius):
        ans_surf = Cylinder(point, axis, radius, transform=transform)
        surf = Cylinder(point, axis, radius)
        surf_tr = surf.transform(transform)
        np.testing.assert_array_almost_equal(ans_surf._pt, surf_tr._pt)
        np.testing.assert_array_almost_equal(ans_surf._axis, surf_tr._axis)
        np.testing.assert_almost_equal(ans_surf._radius, surf_tr._radius)

    @pytest.mark.parametrize('point, axis, radius, options', [
        ([1, 2, 3], np.array([1, 2, 3]) / np.sqrt(14), 5, {}),
        ([1, 2, 3], np.array([1, 2, 3]) / np.sqrt(14), 5, {'name': 1}),
        ([1, 2, 3], np.array([1, 2, 3]) / np.sqrt(14), 5, {'name': 2}),
    ])
    def test_pickle(self, point, axis, radius, options):
        surf = Cylinder(point, axis, radius, **options)
        with open('test.pic', 'bw') as f:
            pickle.dump(surf, f, pickle.HIGHEST_PROTOCOL)
        with open('test.pic', 'br') as f:
            surf_un = pickle.load(f)
        np.testing.assert_array_almost_equal(surf._pt, surf_un._pt)
        np.testing.assert_array_almost_equal(surf._axis, surf_un._axis)
        np.testing.assert_almost_equal(surf._radius, surf_un._radius)
        assert surf.options == surf_un.options


class TestCone:
    @pytest.mark.parametrize('apex, axis, tan2, ap, ax, t2', [
        ([1, 2, 3], [1, 2, 3], 0.5, [1, 2, 3],
         np.array([1, 2, 3]) / np.sqrt(14), 0.25)
    ])
    def test_init(self, transform, apex, axis, tan2, ap, ax, t2):
        surf = Cone(apex, axis, tan2, transform=transform)
        ap = transform.apply2point(ap)
        ax = transform.apply2vector(ax)
        np.testing.assert_array_almost_equal(ap, surf._apex)
        np.testing.assert_array_almost_equal(ax, surf._axis)
        np.testing.assert_array_almost_equal(t2, surf._t2)

    @pytest.mark.parametrize('sheet, case', [(0, 0), (1, 1), (-1, 2)])
    @pytest.mark.parametrize('point, expected', [
        ([0, 1, 2], ([-1], [-1], [+1])),
        ([0, 1, 0.3], ([-1], [-1], [+1])),
        ([0, 1, 0.2], ([+1], [+1], [+1])),
        ([0, 1, 3.7], ([-1], [-1], [+1])),
        ([0, 1, 3.8], ([+1], [+1], [+1])),
        ([0, 2.7, 2], ([-1], [-1], [+1])),
        ([0, 2.8, 2], ([+1], [+1], [+1])),
        ([0, -0.7, 2], ([-1], [-1], [+1])),
        ([0, -0.8, 2], ([+1], [+1], [+1])),
        ([-6, 1, 0.3], ([-1], [+1], [-1])),
        ([-6, 1, 0.2], ([+1], [+1], [+1])),
        ([-6, 1, 3.7], ([-1], [+1], [-1])),
        ([-6, 1, 3.8], ([+1], [+1], [+1])),
        ([-6, 2.7, 2], ([-1], [+1], [-1])),
        ([-6, 2.8, 2], ([+1], [+1], [+1])),
        ([-6, -0.7, 2], ([-1], [+1], [-1])),
        ([-6, -0.8, 2], ([+1], [+1], [+1])),
        ([3, 1, -1.4], ([-1], [-1], [+1])),
        ([3, 1, -1.5], ([+1], [+1], [+1])),
        ([3, 1, 5.4], ([-1], [-1], [+1])),
        ([3, 1, 5.5], ([+1], [+1], [+1])),
        ([3, 4.4, 2], ([-1], [-1], [+1])),
        ([3, 4.5, 2], ([+1], [+1], [+1])),
        ([3, -2.4, 2], ([-1], [-1], [+1])),
        ([3, -2.5, 2], ([+1], [+1], [+1])),
        (np.array([[0, 1, 2], [0, 1, 0.3], [0, 1, 0.2], [0, 1, 3.7],
                   [0, 1, 3.8], [0, 2.7, 2], [0, 2.8, 2], [0, -0.7, 2],
                   [0, -0.8, 2], [-6, 1, 0.3], [-6, 1, 0.2], [-6, 1, 3.7],
                   [-6, 1, 3.8], [-6, 2.7, 2], [-6, 2.8, 2], [-6, -0.7, 2],
                   [-6, -0.8, 2], [3, 1, -1.4], [3, 1, -1.5], [3, 1, 5.4],
                   [3, 1, 5.5], [3, 4.4, 2], [3, 4.5, 2], [3, -2.4, 2],
                   [3, -2.5, 2]]),
         ([-1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1,
          +1, -1, +1, -1, +1, -1, +1, -1, +1],
          [-1, -1, +1, -1, +1, -1, +1, -1, +1, +1, +1, +1, +1, +1, +1,
           +1,
           +1, -1, +1, -1, +1, -1, +1, -1, +1],
          [+1, +1, +1, +1, +1, +1, +1, +1, +1, -1, +1, -1, +1, -1, +1,
           -1,
           +1, +1, +1, +1, +1, +1, +1, +1, +1])
          )
    ])
    def test_point_test(self, sheet, case, point, expected):
        surf = Cone([-3, 1, 2], [1, 0, 0], 1 / np.sqrt(3), sheet)
        result = surf.test_points(point)
        np.testing.assert_array_equal(result, expected[case])

    @pytest.mark.parametrize('params', [
        ([0, 0, 0], [1, 0, 0], 0.5, 0), ([0, 0, 0], [1, 0, 0], 0.5, -1, 0),
        ([0, 0, 0], [1, 0, 0], 0.5, +1, 0),
        ([0, 1.4, 0], [1, 0, 0], 0.5, 0), ([0, 1.4, 0], [1, 0, 0], 0.5, -1, 0),
        ([0, 1.4, 0], [1, 0, 0], 0.5, +1, 0),
        ([0, 1.6, 0], [1, 0, 0], 0.5, +1),
        ([0, 1.6, 0], [1, 0, 0], 0.5, -1, +1),
        ([0, 1.6, 0], [1, 0, 0], 0.5, +1, +1),
        ([0, -1.4, 0], [1, 0, 0], 0.5, 0),
        ([0, -1.4, 0], [1, 0, 0], 0.5, -1, 0),
        ([0, -1.4, 0], [1, 0, 0], 0.5, +1, 0),
        ([0, -1.6, 0], [1, 0, 0], 0.5, +1),
        ([0, -1.6, 0], [1, 0, 0], 0.5, -1, +1),
        ([0, -1.6, 0], [1, 0, 0], 0.5, +1, +1),
        ([-1, 1.9, 0], [1, 0, 0], 0.5, 0),
        ([-1, 1.9, 0], [1, 0, 0], 0.5, -1, +1),
        ([-1, 1.9, 0], [1, 0, 0], 0.5, +1, 0),
        ([1, 2.1, 0], [1, 0, 0], 0.5, +1),
        ([1, 2.1, 0], [1, 0, 0], 0.5, -1, +1),
        ([1, 2.1, 0], [1, 0, 0], 0.5, +1, +1),
        ([3.9, 0, 0], [1, 0, 0], 0.5, -1),
        ([3.9, 0, 0], [1, 0, 0], 0.5, -1, -1),
        ([3.9, 0, 0], [1, 0, 0], 0.5, +1, +1),
        ([-3.9, 0, 0], [1, 0, 0], 0.5, -1),
        ([-3.9, 0, 0], [1, 0, 0], 0.5, -1, +1),
        ([-3.9, 0, 0], [1, 0, 0], 0.5, +1, -1),
        ([0, 0, -3.9], [0, 0, 1], 0.5, -1),
        ([0, 0, -3.9], [0, 0, 1], 0.5, -1, +1),
        ([0, 0, -3.9], [0, 0, 1], 0.5, +1, -1),
        ([0, 0, 3.9], [0, 0, 1], 0.5, -1),
        ([0, 0, 3.9], [0, 0, 1], 0.5, -1, -1),
        ([0, 0, 3.9], [0, 0, 1], 0.5, +1, +1),
        ([0, -3.9, 0], [0, 1, 0], 0.5, -1),
        ([0, -3.9, 0], [0, 1, 0], 0.5, -1, +1),
        ([0, -3.9, 0], [0, 1, 0], 0.5, +1, -1),
        ([3.8, 0, 0], [1, 0, 0], 0.5, 0), ([3.8, 0, 0], [1, 0, 0], 0.5, -1, 0),
        ([3.8, 0, 0], [1, 0, 0], 0.5, +1, +1),
        ([-3.8, 0, 0], [1, 0, 0], 0.5, 0),
        ([-3.8, 0, 0], [1, 0, 0], 0.5, -1, +1),
        ([-3.8, 0, 0], [1, 0, 0], 0.5, +1, 0),
        ([0, 0, -3.8], [0, 0, 1], 0.5, 0),
        ([0, 0, -3.8], [0, 0, 1], 0.5, -1, +1),
        ([0, 0, -3.8], [0, 0, 1], 0.5, +1, 0),
        ([0, 0, 3.8], [0, 0, 1], 0.5, 0), ([0, 0, 3.8], [0, 0, 1], 0.5, -1, 0),
        ([0, 0, 3.8], [0, 0, 1], 0.5, +1, +1),
        ([0, -3.8, 0], [0, 1, 0], 0.5, 0),
        ([0, -3.8, 0], [0, 1, 0], 0.5, -1, +1),
        ([0, -3.8, 0], [0, 1, 0], 0.5, +1, 0)
    ])
    def test_box_test(self, box, params):
        *param, ans = params
        surf = Cone(*param)
        assert surf.test_box(box) == ans

    @pytest.mark.parametrize('apex, axis, t2', [
        ([1, 2, 3],
         np.array([1, 2, 3]) / np.sqrt(14), 0.25),
    ])
    def test_transform(self, transform, apex, axis, t2):
        ans_surf = Cone(apex, axis, t2, transform=transform)
        surf = Cone(apex, axis, t2)
        surf_tr = surf.transform(transform)
        self.assert_cone(ans_surf, surf_tr)

    @staticmethod
    def assert_cone(ans_surf, surf_tr):
        np.testing.assert_array_almost_equal(ans_surf._apex, surf_tr._apex)
        np.testing.assert_array_almost_equal(ans_surf._axis, surf_tr._axis)
        np.testing.assert_almost_equal(ans_surf._t2, surf_tr._t2)
        assert ans_surf._sheet == surf_tr._sheet
        assert ans_surf.options == surf_tr.options

    @pytest.mark.parametrize('apex, axis, t2, sheet, options', [
        ([1, 2, 3],
         np.array([1, 2, 3]) / np.sqrt(14), 0.25, 0, {'name': 1}),
        ([1, 2, 3],
         np.array([1, 2, 3]) / np.sqrt(14), 0.25, 0, {}),
        ([1, 2, 3],
         np.array([1, 2, 3]) / np.sqrt(14), 0.25, -1, {'name': 1}),
        ([1, 2, 3],
         np.array([1, 2, 3]) / np.sqrt(14), 0.25, -1, {}),
        ([1, 2, 3],
         np.array([1, 2, 3]) / np.sqrt(14), 0.25, +1, {'name': 1}),
        ([1, 2, 3],
         np.array([1, 2, 3]) / np.sqrt(14), 0.25, +1, {}),
    ])
    def test_pickle(self, apex, axis, t2, sheet, options):
        surf = Cone(apex, axis, t2, sheet, **options)
        with open('test.pic', 'bw') as f:
            pickle.dump(surf, f, pickle.HIGHEST_PROTOCOL)
        with open('test.pic', 'br') as f:
            surf_un = pickle.load(f)
        self.assert_cone(surf, surf_un)


class TestTorus:
    @pytest.mark.parametrize('center, axis, R, A, B, c, ax, r, a, b', [
        ([1, 2, 3], [0, 0, 1], 4, 2, 1, [1, 2, 3], [0, 0, 1], 4, 2, 1)
    ])
    def test_init(self, transform, center, axis, R, A, B, c, ax, r, a, b):
        surf = Torus(center, axis, R, A, B, transform=transform)
        c = transform.apply2point(c)
        ax = transform.apply2vector(ax)
        np.testing.assert_array_almost_equal(c, surf._center)
        np.testing.assert_array_almost_equal(ax, surf._axis)
        np.testing.assert_array_almost_equal(r, surf._R)
        np.testing.assert_almost_equal(a, surf._a)
        np.testing.assert_almost_equal(b, surf._b)

    @pytest.fixture()
    def torus(self, request):
        return [Torus([0, 0, 0], [1, 0, 0], 4, 2, 1),
                Torus([0, 0, 0], [1, 0, 0], -1, 1, 2),
                Torus([0, 0, 0], [1, 0, 0], 1, 1, 2)]

    @pytest.mark.parametrize('case_no, point, expected', [
        (0, [0, 0, 0], [1]), (0, [0, 0, 2.99], [1]), (0, [0, 0, 5.01], [1]),
        (0, [0, 0, 3.01], [-1]), (0, [0, 0, 4.99], [-1]), (0, [0, 2.99, 0], [1]),
        (0, [0, 5.01, 0], [1]), (0, [0, 3.01, 0], [-1]), (0, [0, 4.99, 0], [-1]),
        (0, [2.01, 0, 4], [1]), (0, [1.99, 0, 4], [-1]), (0, [2.01, 4, 0], [1]),
        (0, [1.99, 4, 0], [-1]),
        (0, np.array([[0, 0, 0], [0, 0, 2.99], [0, 0, 5.01], [0, 0, 3.01],
                   [0, 0, 4.99], [0, 2.99, 0], [0, 5.01, 0], [0, 3.01, 0],
                   [0, 4.99, 0], [2.01, 0, 4], [1.99, 0, 4], [2.01, 4, 0],
                   [1.99, 4, 0]]),
         [1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1]),
        (1, [0, 0, 0], [-1]), (1, [0, 0.999, 0], [-1]), (1, [0, 1.001, 0], [+1]),
        (1, [0, 2.999, 0], [+1]), (1, [0, 3.001, 0], [+1]),
        (2, [0, 0, 0], [-1]), (2, [0, 0.999, 0], [-1]), (2, [0, 1.001, 0], [-1]),
        (2, [0, 2.999, 0], [-1]), (2, [0, 3.001, 0], [+1])
    ])
    def test_point_test(self, torus, case_no, point, expected):
        surf = torus[case_no]
        result = surf.test_points(point)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize('point, axis, radius, a, b, ans', [
        ([0, 0, 0], [1, 0, 0], 1, 0.5, 0.5, 0),
        ([0, 0, 0], [1, 0, 0], 1.91, 0.5, 0.5, 0),
        ([0, 0, 0], [1, 0, 0], 1.92, 0.5, 0.5, +1),
        ([1.4, 0, 0], [1, 0, 0], 1, 0.5, 0.5, 0),
        ([1.6, 0, 0], [1, 0, 0], 1, 0.5, 0.5, +1),
        ([0.5, 3, 0], [1, 0, 0], 1.6, 0.5, 0.5, 0),
        ([0.5, 3, 0], [1, 0, 0], 1.4, 0.5, 0.5, +1),
        ([0, 3, 0], [1, 0, 0], 3, 1.4, 1.4, 0),
        ([0, 3, 0], [1, 0, 0], 3, 1.55, 1.55, -1),
        ([0, 0, 0], [1, 0, 0], 1, 1.4, 1.4, 0),
        ([0, 0, 0], [1, 0, 0], 1, 1.5, 1.5, -1),
        ([0, 0, 0], [1, 0, 0], -1, 1.4, 1.4, 0),
        ([0, 0, 0], [1, 0, 0], -1, 2.6, 2.6, 0),
        ([0, 0, 0], [1, 0, 0], -1, 2.7, 2.7, -1),
        ([0, 0, 0], [1, 0, 0], 1.41, 1.42, 2, -1),
        ([0, 0, 0], [1, 0, 0], 1.41, 1.40, 2, 0),
        ([0, 0, 0], [0, 1, 1], 2.2, 0.5, 0.5, 0),
        ([0, 0, 0], [0, 1, 1], 2.3, 0.5, 0.5, +1),
        ([0, 0, 0], [1, 1, 1], 1, 1.9, 1.9, 0),
        ([0, 0, 0], [1, 1, 1], 1, 2.1, 2.1, -1),
        ([0, 0, 0], [0, 1, 0], 1, 0.5, 0.5, 0),
        ([0, 0, 0], [0, 1, 0], 1.91, 0.5, 0.5, 0),
        ([0, 0, 0], [0, 1, 0], 1.92, 0.5, 0.5, +1),
        ([0, 1.4, 0], [0, 1, 0], 1, 0.5, 0.5, 0),
        ([0, 1.6, 0], [0, 1, 0], 1, 0.5, 0.5, +1),
        ([3, 0.5, 0], [0, 1, 0], 1.6, 0.5, 0.5, 0),
        ([3, 0.5, 0], [0, 1, 0], 1.4, 0.5, 0.5, +1),
        ([3, 0, 0], [0, 1, 0], 3, 1.4, 1.4, 0),
        ([3, 0, 0], [0, 1, 0], 3, 1.55, 1.55, -1),
        ([0, 0, 0], [0, 1, 0], 1, 1.4, 1.4, 0),
        ([0, 0, 0], [0, 1, 0], 1, 1.5, 1.5, -1)
    ])
    def test_box_test(self, box, point, axis, radius, a, b, ans):
        surf = Torus(point, axis, radius, a, b)
        assert surf.test_box(box) == ans

    @pytest.mark.parametrize('point, axis, radius, a, b', [
        ([1, 2, 3], [0, 0, 1], 4, 2, 1),
    ])
    def test_transform(self, transform, point, axis, radius, a, b):
        ans_surf = Torus(point, axis, radius, a, b, transform=transform)
        surf = Torus(point, axis, radius, a, b)
        surf_tr = surf.transform(transform)
        self.assert_torus(ans_surf, surf_tr)

    @staticmethod
    def assert_torus(ans_surf, surf_tr):
        np.testing.assert_array_almost_equal(ans_surf._center, surf_tr._center)
        np.testing.assert_array_almost_equal(ans_surf._axis, surf_tr._axis)
        np.testing.assert_almost_equal(ans_surf._R, surf_tr._R)
        np.testing.assert_almost_equal(ans_surf._a, surf_tr._a)
        np.testing.assert_almost_equal(ans_surf._b, surf_tr._b)
        assert surf_tr.options == ans_surf.options

    @pytest.mark.parametrize('point, axis, radius, a, b, options', [
        ([1, 2, 3], [0, 0, 1], 4, 2, 1, {}),
        ([1, 2, 3], [0, 0, 1], 4, 2, 1, {'name': 4}),
    ])
    def test_pickle(self, point, axis, radius, a, b, options):
        surf = Torus(point, axis, radius, a, b, **options)
        with open('test.pic', 'bw') as f:
            pickle.dump(surf, f, pickle.HIGHEST_PROTOCOL)
        with open('test.pic', 'br') as f:
            surf_un = pickle.load(f)
        self.assert_torus(surf, surf_un)


class TestGQuadratic:
    @pytest.mark.parametrize('m, v, k, _m, _v, _k', [
        ([[1, 0, 0], [0, 2, 0], [0, 0, 3]], [1, 2, 3], -4, np.diag([1, 2, 3]),
         [1, 2, 3], -4)
    ])
    def test_init(self, transform, m, v, k, _m, _v, _k):
        surf = GQuadratic(m, v, k, transform=transform)
        _m, _v, _k = transform.apply2gq(_m, _v, _k)
        np.testing.assert_array_almost_equal(_m, surf._m)
        np.testing.assert_array_almost_equal(_v, surf._v)
        np.testing.assert_array_almost_equal(_k, surf._k)

    @pytest.mark.parametrize('point, expected', [
        (np.array([0, 0, 0]), [-1]), (np.array([-0.999, 0, 0]), [-1]),
        (np.array([0.999, 0, 0]), [-1]), (np.array([-1.001, 0, 0]), [+1]),
        (np.array([1.001, 0, 0]), [+1]), (np.array([0, 0.999, 0]), [-1]),
        (np.array([0, 1.001, 0]), [+1]), (np.array([0, 0, -0.999]), [-1]),
        (np.array([[0, 0, 0], [-0.999, 0, 0], [0.999, 0, 0], [-1.001, 0, 0],
                   [1.001, 0, 0], [0, 0.999, 0], [0, 1.001, 0],
                   [0, 0, -0.999]]),
         np.array([-1, -1, -1, 1, 1, -1, 1, -1]))
    ])
    def test_point_test(self, point, expected):
        surf = GQuadratic(np.diag([1, 1, 1]), [0, 0, 0], -1)
        result = surf.test_points(point)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize('m, v, k, ans', [
        (np.diag([1, 1, 1]), [0, 0, 0], -1, 0),
        (np.diag([1, 1, 1]), [0, 0, 0], -0.1, 0),
        (np.diag([1, 1, 1]), [0, 0, 0], -3.01, -1),
        (np.diag([1, 1, 1]), -2 * np.array([1, 1, 1]), 3 - 0.1, 0),
        (np.diag([1, 1, 1]), -2 * np.array([2, 2, 2]), 12 - 3.01, 0),
        (np.diag([1, 1, 1]), -2 * np.array([2, 2, 2]), 12 - 2.99, +1),
        (np.diag([1, 1, 1]), -2 * np.array([2, 0, 0]), 4 - 1.01, 0),
        (np.diag([1, 1, 1]), -2 * np.array([2, 0, 0]), 4 - 0.99, +1),
        (np.diag([1, 1, 1]), -2 * np.array([100, 0, 100]), 20000 - 2, +1)
    ])
    def test_box_test(self, box, m, v, k, ans):
        surf = GQuadratic(m, v, k)
        assert surf.test_box(box) == ans

    @pytest.mark.parametrize('m, v, k', [
        (np.diag([1, 2, 3]), [1, 2, 3], -4),
    ])
    def test_transform(self, transform, m, v, k):
        ans_surf = GQuadratic(m, v, k, transform=transform)
        surf = GQuadratic(m, v, k)
        surf_tr = surf.transform(transform)
        self.assert_gq(ans_surf, surf_tr)

    @staticmethod
    def assert_gq(ans_surf, surf_tr):
        np.testing.assert_array_almost_equal(ans_surf._m, surf_tr._m)
        np.testing.assert_array_almost_equal(ans_surf._v, surf_tr._v)
        np.testing.assert_almost_equal(ans_surf._k, surf_tr._k)
        assert ans_surf.options == surf_tr.options

    @pytest.mark.parametrize('m, v, k, options', [
        (np.diag([1, 2, 3]), [1, 2, 3], -4, {}),
        (np.diag([1, 2, 3]), [1, 2, 3], -4, {'name': 5}),
    ])
    def test_pickle(self, m, v, k, options):
        surf = GQuadratic(m, v, k, **options)
        with open('test.pic', 'bw') as f:
            pickle.dump(surf, f, pickle.HIGHEST_PROTOCOL)
        with open('test.pic', 'br') as f:
            surf_un = pickle.load(f)
        self.assert_gq(surf, surf_un)
