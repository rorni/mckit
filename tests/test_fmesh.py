import pytest
import numpy as np
from itertools import product

from mckit.fmesh import RectMesh
from mckit.transformation import Transformation
from mckit.geometry import EX, EY, EZ
from mckit.surface import create_surface
from mckit.body import Body
from mckit.material import Material


transforms = [
    None,
    Transformation(translation=[2, -1, 3]),
    Transformation(translation=[1, 2, -3], indegrees=True,
                   rotation=[30, 60, 90, 120, 30, 90, 90, 90, 0]),
    Transformation(translation=[1, 0, 0], rotation=[90, 0, 90, 180, 90, 90,
                                                    90, 90, 0],
                   indegrees=True)
]


bins = [
    {'xbins': [0, 1, 2, 4, 6], 'ybins': [0, 2, 5], 'zbins': [-1, 1, 3]},
    {'xbins': [2, 4, 6, 8], 'ybins': [2, 3], 'zbins': [1, 4, 7]},
    {'xbins': [-3, -1, 1], 'ybins': [-2, -1, 0, 1], 'zbins': [-5, -4, -3]}
]


def create_rmesh(bins, tr):
    return RectMesh(bins['xbins'], bins['ybins'], bins['zbins'], transform=tr)


class TestRectMesh:
    @pytest.mark.parametrize('mi, ti', [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3)
    ])
    def test_creation(self, mi, ti):
        bin = bins[mi]
        tr = transforms[ti]
        mesh = create_rmesh(bin, tr)
        np.testing.assert_array_almost_equal(mesh._xbins, bin['xbins'])
        np.testing.assert_array_almost_equal(mesh._ybins, bin['ybins'])
        np.testing.assert_array_almost_equal(mesh._zbins, bin['zbins'])

        origin = [bin['xbins'][0], bin['ybins'][0], bin['zbins'][0]]
        ex = EX
        ey = EY
        ez = EZ
        if tr is not None:
            ex = tr.apply2vector(ex)
            ey = tr.apply2vector(ey)
            ez = tr.apply2vector(ez)
            origin = tr.apply2point(origin)
        np.testing.assert_array_almost_equal(mesh._origin, origin)
        np.testing.assert_array_almost_equal(mesh._ex, ex)
        np.testing.assert_array_almost_equal(mesh._ey, ey)
        np.testing.assert_array_almost_equal(mesh._ez, ez)

    @pytest.mark.parametrize('mi, ti, ans', [
        (0, 0, (4, 2, 2)), (0, 1, (4, 2, 2)), (0, 2, (4, 2, 2)),
        (0, 3, (4, 2, 2)),
        (1, 0, (3, 1, 2)), (1, 1, (3, 1, 2)), (1, 2, (3, 1, 2)),
        (1, 3, (3, 1, 2)),
        (2, 0, (2, 3, 2)), (2, 1, (2, 3, 2)), (2, 2, (2, 3, 2)),
        (2, 3, (2, 3, 2))
    ])
    def test_shape(self, mi, ti, ans):
        mesh = create_rmesh(bins[mi], transforms[ti])
        assert mesh.shape == ans

    surfaces = {
        1: create_surface('PY', 3.5),
        2: create_surface('PY', -1.0),
        3: create_surface('PX', 3.0),
        4: create_surface('PX', 1.5),
        5: create_surface('PZ', 2.0),
        6: create_surface('PZ', -2.0),
        7: create_surface('PX', 5.0)
    }

    bodies = [
        Body([surfaces[1], 'C', surfaces[2], 'I', surfaces[3], 'C', 'I',
              surfaces[4], 'I', surfaces[5], 'C', 'I', surfaces[6], 'I'],
             name=1, MAT=Material(atomic=[('Fe', 1)], density=7.8)),
        Body([surfaces[1], 'C', surfaces[2], 'I', surfaces[7], 'C', 'I',
              surfaces[3], 'I', surfaces[5], 'C', 'I', surfaces[6], 'I'],
             name=2, MAT=Material(atomic=[('C', 1)], density=2.7)),
        Body([surfaces[1], surfaces[2], 'C', 'U', surfaces[4], 'C', 'U',
              surfaces[7], 'U', surfaces[5], 'U', surfaces[6], 'C', 'U'],
             name=3)
    ]

    @pytest.mark.parametrize('mi, wmo, expected', [
        (0, True, {
            bodies[0]: np.array([
                [[0, 0], [0, 0]], [[2, 1], [1.5, 0.75]], [[4, 2], [3, 1.5]],
                [[0, 0], [0, 0]]
            ]),
            bodies[1]: np.array([
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[4, 2], [3, 1.5]],
                [[4, 2], [3, 1.5]]
            ])
        }),
        (0, False, {
            bodies[0]: np.array([
                [[0, 0], [0, 0]], [[2, 1], [1.5, 0.75]], [[4, 2], [3, 1.5]],
                [[0, 0], [0, 0]]
            ]),
            bodies[1]: np.array([
                [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[4, 2], [3, 1.5]],
                [[4, 2], [3, 1.5]]
            ]),
            bodies[2]: np.array([
                [[4, 4], [6, 6]], [[2, 3], [4.5, 5.25]], [[0, 4.0], [6.0, 9.0]],
                [[4.0, 6], [9, 10.5]]
            ])
        })
    ])
    def test_calculate_volumes(self, mi, wmo, expected):
        mesh = create_rmesh(bins[mi], None)
        volumes = mesh.calculate_volumes(self.bodies, with_mat_only=wmo, min_volume=1.e-5)
        assert volumes.keys() == expected.keys()
        for bd, vols in expected.items():
            for i, x in enumerate(vols):
                for j, y in enumerate(x):
                    for k, v in enumerate(y):
                        assert volumes[bd][i, j, k] == pytest.approx(v, rel=5.e-2)

    @pytest.mark.parametrize('mi, ti', product(range(len(bins)), range(len(transforms))))
    def test_get_voxel(self, mi, ti):
        tr = transforms[ti]
        bin = bins[mi]
        mesh = create_rmesh(bin, tr)
        shape = mesh.shape
        ex = EX
        ey = EY
        ez = EZ
        for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
            vox = mesh.get_voxel(i, j, k)
            corners = []
            for pr in product(bin['xbins'][i:i+2], bin['ybins'][j:j+2],
                              bin['zbins'][k:k+2]):
                corners.append(list(pr))
            if tr is not None:
                corners = tr.apply2point(corners)
            np.testing.assert_array_almost_equal(vox.corners, corners)


    points = [
        [0.5, 0.9, 0], [-1.4, 0.5, 0.1], [3, 2.5, 1.9], [0.5, 0.1, -0.3],
        [-1.8, -1.5, -3.5], [-0.1, 0.4, -4.1], [5.1, 4.9, 1.4],
        [[0.5, 0.9, 0], [-1.4, 0.5, 0.1], [3, 2.5, 1.9], [0.5, 0.1, -0.3],
         [-1.8, -1.5, -3.5], [-0.1, 0.4, -4.1], [5.1, 4.9, 1.4]]
    ]

    @pytest.mark.parametrize('mi, ti, pi, local',
                             product(range(len(bins)), range(len(transforms)),
                                     range(len(points)), [False, True]))
    def test_voxel_index(self, mi, ti, pi, local):
        tr = transforms[ti]
        bin = bins[mi]
        mesh = create_rmesh(bin, tr)
        pt = self.points[pi]
        result = mesh.voxel_index(pt, local=local)
        if local == False and tr is not None:
            pt = tr.reverse().apply2point(pt)

        def check_one(r, pt):
            if r is not None:
                print(r)
                i, j, k = r
                assert bin['xbins'][i] <= pt[0] <= bin['xbins'][i+1]
                assert bin['ybins'][j] <= pt[1] <= bin['ybins'][j+1]
                assert bin['zbins'][k] <= pt[2] <= bin['zbins'][k+1]
            else:
                px = pt[0] <= bin['xbins'][0] or pt[0] >= bin['xbins'][-1]
                py = pt[1] <= bin['ybins'][0] or pt[1] >= bin['ybins'][-1]
                pz = pt[2] <= bin['zbins'][0] or pt[2] >= bin['zbins'][-1]
                assert px or py or pz

        if np.array(pt).shape == (3,):
            check_one(result, pt)
        else:
            for r, p in zip(result, pt):
                check_one(r, p)

    @pytest.mark.parametrize('mi, index, answer', [
        (0, (3, 1, 0), (3, 1, 0)),
        (0, (0, 0, 0), (0, 0, 0)),
        (0, (-1, 0, 0), None),
        (0, (0, -1, 0), None),
        (0, (0, 0, -1), None),
        (0, (4, 1, 1), None),
        (0, (3, 2, 1), None),
        (0, (3, 1, 2), None),
        (1, (2, 0, 1), (2, 0, 1)),
        (1, (2, 1, 2), None),
        (2, (1, 2, 1), (1, 2, 1)),
        (2, (1, 3, 1), None),
        (2, (10, 10, 10), None),
        (2, (10, -10, 10), None)
    ])
    def test_check_indices(self, mi, index, answer):
        mesh = create_rmesh(bins[mi], None)
        check = mesh.check_indices(*index)
        assert check == answer

    @pytest.mark.parametrize('ti', range(len(transforms)))
    @pytest.mark.parametrize('mi, args, expected', [
        (0, {}, None),
        (0, {'X': 0.5}, {'axis': 0, 'index': 0, 'x': [1.0, 3.5], 'y': [0.0, 2.0]}),
        (0, {'Y': 0.5}, {'axis': 1, 'index': 0, 'x': [0.5, 1.5, 3.0, 5.0], 'y': [0.0, 2.0]}),
        (0, {'Z': 0.5}, {'axis': 2, 'index': 0, 'x': [0.5, 1.5, 3.0, 5.0], 'y': [1.0, 3.5]}),
        (0, {'X': -1}, None),
        (1, {'Y': 10}, None),
        (2, {'Z': -100}, None),
        (0, {'X': 0.5, 'Y': 0.5, 'Z': 0.5}, None),
        (1, {'X': 0, 'Y': 2.5}, None),
        (1, {'X': 3, 'Y': 0}, None),
        (1, {'X': 3, 'Z': 9}, None),
        (2, {'X': -2, 'Y': 0.5}, None),
        (1, {'Y': 2.9}, {'axis': 1, 'index': 0, 'x': [3.0, 5.0, 7.0], 'y': [2.5, 5.5]}),
        (1, {'Z': 4.9}, {'axis': 2, 'index': 1, 'x': [3.0, 5.0, 7.0], 'y': [2.5]}),
        (2, {'X': 0.0}, {'axis': 0, 'index': 1, 'x': [-1.5, -0.5, 0.5], 'y': [-4.5, -3.5]}),
        (2, {'Z': -4.1}, {'axis': 2, 'index': 0, 'x': [-2.0, 0.0], 'y': [-1.5, -0.5, 0.5]}),
        (0, {'X': 5}, {'axis': 0, 'index': 3, 'x': [1.0, 3.5], 'y': [0.0, 2.0]})
    ])
    def test_slice_index(self, ti, mi, args, expected):
        tr = transforms[ti]
        bin = bins[mi]
        mesh = create_rmesh(bin, tr)
        if expected is None:
            with pytest.raises(ValueError):
                mesh.slice_axis_index(**args)
        else:
            axis, index, x, y = mesh.slice_axis_index(**args)
            assert axis == expected['axis']
            assert index == expected['index']
            np.testing.assert_array_almost_equal(x, expected['x'])
            np.testing.assert_array_almost_equal(y, expected['y'])

