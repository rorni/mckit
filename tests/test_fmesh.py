import unittest

import numpy.testing as npt
import numpy as np

from mckit.fmesh import Box

from tests.fmesh_test_data import box_test_data as btd


class TestBox(unittest.TestCase):
    def test_test_point(self):
        for i, bd in enumerate(btd.box_data):
            box = Box(bd['box']['basis'], bd['box']['ex'], bd['box']['ey'],
                      bd['box']['ez'])
            for j, pt in enumerate(btd.points):
                with self.subTest(msg='box {0} point {1}'.format(i, j)):
                    ans = box.test_point(pt)
                    if isinstance(bd['points'][j], bool):
                        self.assertEqual(ans, bd['points'][j])
                    else:
                        print(ans)
                        self.assertListEqual(list(ans), bd['points'][j])

    def test_split(self):
        for i, bd in enumerate(btd.box_data):
            box = Box(bd['box']['basis'], bd['box']['ex'], bd['box']['ey'],
                      bd['box']['ez'])
            for j, params in enumerate(btd.splits):
                with self.subTest(msg='box {0}, split {1}'.format(i, j)):
                    box1, box2 = box.split(**params)
                    npt.assert_array_almost_equal(box1.base,
                                         bd['split'][j][0]['basis'])
                    npt.assert_array_almost_equal(box2.base,
                                         bd['split'][j][1]['basis'])
                    npt.assert_array_almost_equal(box1.ex,
                                                  bd['split'][j][0]['ex'])
                    npt.assert_array_almost_equal(box2.ex,
                                                  bd['split'][j][1]['ex'])
                    npt.assert_array_almost_equal(box1.ey,
                                                  bd['split'][j][0]['ey'])
                    npt.assert_array_almost_equal(box2.ey,
                                                  bd['split'][j][1]['ey'])
                    npt.assert_array_almost_equal(box1.ez,
                                                  bd['split'][j][0]['ez'])
                    npt.assert_array_almost_equal(box2.ez,
                                                  bd['split'][j][1]['ez'])

    def test_corners(self):
        for i, bd in enumerate(btd.box_data):
            box = Box(bd['box']['basis'], bd['box']['ex'], bd['box']['ey'],
                      bd['box']['ez'])
            with self.subTest(msg='box {0}'.format(i)):
                corners = box.corners()
                corners = [list(corners[j, :]) for j in range(corners.shape[0])]
                corners = sorted(corners)
                self.assertListEqual(corners, sorted(bd['corners']))

    def test_bounds(self):
        for i, bd in enumerate(btd.box_data):
            box = Box(bd['box']['basis'], bd['box']['ex'], bd['box']['ey'],
                      bd['box']['ez'])
            with self.subTest(msg='box {0}'.format(i)):
                bounds = box.bounds()
                self.assertListEqual(bounds, bd['bounds'])

    def test_volume(self):
        for i, bd in enumerate(btd.box_data):
            box = Box(bd['box']['basis'], bd['box']['ex'], bd['box']['ey'],
                      bd['box']['ez'])
            with self.subTest(msg='box {0}'.format(i)):
                vol = box.volume()
                self.assertAlmostEqual(vol, bd['volume'])

    def test_generate_random_points(self):
        for i, bd in enumerate(btd.box_data):
            box = Box(bd['box']['basis'], bd['box']['ex'], bd['box']['ey'],
                      bd['box']['ez'])
            with self.subTest(msg='box {0}'.format(i)):
                points = box.generate_random_points(100)
                pt = box.test_point(points)
                self.assertEqual(np.all(pt), True)
                retrieve = box.get_random_points()
                npt.assert_array_almost_equal(points, retrieve)


if __name__ == '__main__':
    unittest.main()
