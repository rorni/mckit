import unittest

from mckit.parser.mcnp_input_parser import read_mcnp
from mckit.fmesh import Box

from tests.universe_test_data.volume_ans import *


@unittest.skip
class TestUniverse(unittest.TestCase):
    def test_box_volumes(self):
        for case in case_names:
            model = read_mcnp('tests/universe_test_data/{0}.txt'.format(case))
            universe = model.universe('uvnc sample')
            for box_data, (ans_volumes, errors) in volume_ans[case].items():
                msg = "case: {0}, box: {1}".format(case, box_data)
                with self.subTest(msg=msg):
                    base = [box_data[0][0], box_data[1][0], box_data[2][0]]
                    ex = [box_data[0][1] - box_data[0][0], 0, 0]
                    ey = [0, box_data[1][1] - box_data[1][0], 0]
                    ez = [0, 0, box_data[2][1] - box_data[2][0]]
                    box = Box(base, ex, ey, ez)
                    volume_dict = universe.get_box_volumes(
                        box, accuracy=1, names=range(len(ans_volumes)),
                        verbose=True, pool_size=100000
                    )
                    volumes = np.zeros((len(ans_volumes),))
                    for i, v in volume_dict.items():
                        volumes[i] = v
                    deltas = ans_volumes * errors
                    for va, err, vc, d in zip(ans_volumes, errors, volumes, deltas):
                        delta = max(0.2**3, d) * 4
                        self.assertAlmostEqual(va, vc, delta=delta)


if __name__ == '__main__':
    unittest.main()
