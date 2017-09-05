from mckit.material import Material

materials = [
    Material(atomic=[(1000, 2), (8000, 1)], density=0.9982),
    Material(atomic=[(1000, 2), (8000, 1)], density=0.1),
    Material(atomic=[('Si', 1), ('O', 2)], density=2.65),
    Material(wgt=[('N', 75.5), ('O', 23.15), ('Ar', 1.292)], density=1.247e-3)
]

densities = [
    (-0.9982, materials[0]), (-0.1, materials[1]), (-2.0, None),
    (-0.9984, None), (-0.9980, None), (1.0010337342849073e+23, materials[0]),
    (1.0028388442044753e+22, materials[1]), (1.0012337342849073e+23, None),
    (1.0098337342849073e+23, None), (-0.99825, materials[0]),
    (-0.99815, materials[0]), (-2.65, materials[2])
]

compositions = {
    1: {'atomic': [(1000, 2), (8000, 1)]},
    2:
}

