import io
import pickle

import numpy as np
import pytest

from mckit.box import Box

# noinspection PyUnresolvedReferences,PyPackageRequirements
from mckit.geometry import EX, EY, EZ


@pytest.fixture(scope="module")
def box():
    boxes = [Box([0.5, 1, 1.5], 1, 2, 3), Box([0.5, -1, 1.5], 3, 2, 1)]
    return boxes


@pytest.mark.parametrize(
    "point_no, point",
    enumerate(
        [
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [-0.1, -0.1, 1.1],
            [0.1, 0.1, 3.2],
            [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [-0.1, -0.1, 1.1], [0.1, 0.1, 3.2]],
        ]
    ),
)
@pytest.mark.parametrize(
    "case_no, answer",
    enumerate(
        [
            {0: True, 1: True, 2: False, 3: False, 4: [True, True, False, False]},
            {0: False, 1: False, 2: True, 3: False, 4: [False, False, True, False]},
        ]
    ),
)
def test_test_point(box, point_no, point, case_no, answer):
    b = box[case_no]
    result = b.test_points(point)
    expected = answer[point_no]
    assert np.all(result == expected)


@pytest.fixture(scope="module")
def splits():
    return [
        {"dir": "x", "ratio": 0.5},
        {"dir": "y", "ratio": 0.5},
        {"dir": "z", "ratio": 0.5},
        {"ratio": 0.5},
        {"dir": "x", "ratio": 0.2},
        {"dir": "y", "ratio": 0.2},
        {"dir": "z", "ratio": 0.2},
        {"ratio": 0.2},
    ]


@pytest.mark.parametrize(
    "case_no, split_no, ba1, ba2",
    [
        (
            0,
            0,
            {
                "basis": [0.25, 1, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 0.5,
                "ydim": 2,
                "zdim": 3,
            },
            {
                "basis": [0.75, 1, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 0.5,
                "ydim": 2,
                "zdim": 3,
            },
        ),
        (
            0,
            1,
            {
                "basis": [0.5, 0.5, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 3],
                "xdim": 1,
                "ydim": 1,
                "zdim": 3,
            },
            {
                "basis": [0.5, 1.5, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1,
                "ydim": 1,
                "zdim": 3,
            },
        ),
        (
            0,
            2,
            {
                "basis": [0.5, 1, 0.75],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1,
                "ydim": 2,
                "zdim": 1.5,
            },
            {
                "basis": [0.5, 1, 2.25],
                "ex": [1, 0, 0],
                "ey": [0, 2, 0],
                "ez": [0, 0, 1.5],
                "xdim": 1,
                "ydim": 2,
                "zdim": 1.5,
            },
        ),
        (
            0,
            3,
            {
                "basis": [0.5, 1, 0.75],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1,
                "ydim": 2,
                "zdim": 1.5,
            },
            {
                "basis": [0.5, 1, 2.25],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1,
                "ydim": 2,
                "zdim": 1.5,
            },
        ),
        (
            0,
            4,
            {
                "basis": [0.1, 1, 1.5],
                "ex": [0.2, 0, 0],
                "ey": [0, 2, 0],
                "ez": [0, 0, 3],
                "xdim": 0.2,
                "ydim": 2,
                "zdim": 3,
            },
            {
                "basis": [0.6, 1, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 0.8,
                "ydim": 2,
                "zdim": 3,
            },
        ),
        (
            0,
            5,
            {
                "basis": [0.5, 0.2, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1,
                "ydim": 0.4,
                "zdim": 3,
            },
            {
                "basis": [0.5, 1.2, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1,
                "ydim": 1.6,
                "zdim": 3,
            },
        ),
        (
            0,
            6,
            {
                "basis": [0.5, 1, 0.3],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1,
                "ydim": 2,
                "zdim": 0.6,
            },
            {
                "basis": [0.5, 1, 1.8],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1,
                "ydim": 2,
                "zdim": 2.4,
            },
        ),
        (
            0,
            7,
            {
                "basis": [0.5, 1, 0.3],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1,
                "ydim": 2,
                "zdim": 0.6,
            },
            {
                "basis": [0.5, 1, 1.8],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1,
                "ydim": 2,
                "zdim": 2.4,
            },
        ),
        (
            1,
            0,
            {
                "basis": [-0.25, -1, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1.5,
                "ydim": 2,
                "zdim": 1,
            },
            {
                "basis": [1.25, -1, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1.5,
                "ydim": 2,
                "zdim": 1,
            },
        ),
        (
            1,
            1,
            {
                "basis": [0.5, -1.5, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 3,
                "ydim": 1,
                "zdim": 1,
            },
            {
                "basis": [0.5, -0.5, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 3,
                "ydim": 1,
                "zdim": 1,
            },
        ),
        (
            1,
            2,
            {
                "basis": [0.5, -1, 1.25],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 3,
                "ydim": 2,
                "zdim": 0.5,
            },
            {
                "basis": [0.5, -1, 1.75],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 3,
                "ydim": 2,
                "zdim": 0.5,
            },
        ),
        (
            1,
            3,
            {
                "basis": [-0.25, -1, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1.5,
                "ydim": 2,
                "zdim": 1,
            },
            {
                "basis": [1.25, -1, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 1.5,
                "ydim": 2,
                "zdim": 1,
            },
        ),
        (
            1,
            4,
            {
                "basis": [-0.7, -1, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 0.6,
                "ydim": 2,
                "zdim": 1,
            },
            {
                "basis": [0.8, -1, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 2.4,
                "ydim": 2,
                "zdim": 1,
            },
        ),
        (
            1,
            5,
            {
                "basis": [0.5, -1.8, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 3,
                "ydim": 0.4,
                "zdim": 1,
            },
            {
                "basis": [0.5, -0.8, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 3,
                "ydim": 1.6,
                "zdim": 1,
            },
        ),
        (
            1,
            6,
            {
                "basis": [0.5, -1, 1.1],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 3,
                "ydim": 1,
                "zdim": 0.2,
            },
            {
                "basis": [0.5, -1, 1.6],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 3,
                "ydim": 2,
                "zdim": 0.8,
            },
        ),
        (
            1,
            7,
            {
                "basis": [-0.7, -1, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 0.6,
                "ydim": 2,
                "zdim": 1,
            },
            {
                "basis": [0.8, -1, 1.5],
                "ex": [1, 0, 0],
                "ey": [0, 1, 0],
                "ez": [0, 0, 1],
                "xdim": 2.4,
                "ydim": 2,
                "zdim": 1,
            },
        ),
    ],
)
def test_split(box, splits, case_no, split_no, ba1, ba2):
    box1, box2 = box[case_no].split(**splits[split_no])
    np.testing.assert_array_almost_equal(box1.center, ba1["basis"])
    np.testing.assert_array_almost_equal(box2.center, ba2["basis"])


@pytest.mark.parametrize(
    "case_no, expected",
    enumerate(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 3.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 3.0],
                [1.0, 2.0, 0.0],
                [1.0, 2.0, 3.0],
            ],
            [
                [-1.0, -2.0, 1.0],
                [-1.0, -2.0, 2.0],
                [-1.0, 0.0, 1.0],
                [-1.0, 0.0, 2.0],
                [2.0, -2.0, 1.0],
                [2.0, -2.0, 2.0],
                [2.0, 0.0, 1.0],
                [2.0, 0.0, 2.0],
            ],
        ]
    ),
)
def test_corners(box, case_no, expected):
    corners = box[case_no].corners
    assert np.all(corners == expected)


@pytest.mark.parametrize(
    "case_no, expected",
    enumerate([[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], [[-1.0, 2.0], [-2.0, 0.0], [1.0, 2.0]]]),
)
def test_bounds(box, case_no, expected):
    bounds = box[case_no].bounds
    assert np.all(bounds == expected)


@pytest.mark.parametrize("case_no, expected", enumerate([6, 6]))
def test_volume(box, case_no, expected):
    vol = box[case_no].volume
    assert vol == expected


@pytest.mark.parametrize("case_no", range(2))
def test_random_points(box, case_no):
    points = box[case_no].generate_random_points(100)
    pt = box[case_no].test_points(points)
    assert np.all(pt) == True


boxes = [
    Box([0, 0, 0], 1, 1, 1),
    Box([2, 0, 0], 0.5, 4, 2),
    Box([0, 0, 2], 0.5, 4, 2),
    Box([0, 0, 2], 0.2, 0.2, 10),
    Box([1, 1, 1], 1.1, 1.1, 1.1),
    Box([-1, -1, -1], 1.1, 1.1, 1.1),
]

eq_matrix = [
    [1, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1],
]


@pytest.mark.parametrize("case1", range(len(boxes)))
@pytest.mark.parametrize("case2", range(len(boxes)))
def test_check_intersection(case1, case2):
    box1 = boxes[case1]
    box2 = boxes[case2]
    answer = bool(eq_matrix[case1][case2])

    result = box1.check_intersection(box2)
    assert result == answer
    result = box2.check_intersection(box1)
    assert result == answer


@pytest.mark.parametrize(
    "center, wx, wy, wz, ex, ey, ez", [([0.0, 0.0, 0.0], 1.0, 2.0, 3.0, EX, EY, EZ)]
)
def test_pickle(center, wx, wy, wz, ex, ey, ez):
    box = Box(center, wx, wy, wz, ex, ey, ez)
    with io.BytesIO() as f:
        pickle.dump(box, f)
        f.seek(0)
        box_unpickled = pickle.load(f)
    assert box == box_unpickled
