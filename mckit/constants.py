# -*- coding: utf-8 -*-

import numpy as np


__all__ = [
    'AVOGADRO',
    'EX', 'EY', 'EZ',
    'ORIGIN', 'IDENTITY_ROTATION',
    'ANGLE_TOLERANCE', 'RESOLUTION',
    'CHARGE_TO_NAME', 'NAME_TO_CHARGE', 'NATURAL_ABUNDANCE', 'ISOTOPE_MASS'
]


AVOGADRO = 6.02214085774e+23

# basis vectors
EX = np.array([1, 0, 0])
EY = np.array([0, 1, 0])
EZ = np.array([0, 0, 1])

# the origin of coordinate system - vector of zeros.
ORIGIN = np.zeros((3,))

# identity rotation matrix
IDENTITY_ROTATION = np.eye(3)

# angle tolerance
ANGLE_TOLERANCE = 0.001

# Resolution of float number
RESOLUTION = np.finfo(float).resolution

# Natural presence of isotopes
CHARGE_TO_NAME = {}
NAME_TO_CHARGE = {}
NATURAL_ABUNDANCE = {}
ISOTOPE_MASS = {}

with open('mckit/data/isotopes.dat') as f:
    for line in f:
        number, name, *data = line.split()
        number = int(number)
        name = name.upper()
        CHARGE_TO_NAME[number] = name
        NAME_TO_CHARGE[name] = number
        NATURAL_ABUNDANCE[number] = {}
        ISOTOPE_MASS[number] = {}
        for i in range(len(data) // 3):
            isotope = int(data[i * 3])
            ISOTOPE_MASS[number][isotope] = float(data[i * 3 + 1])
            abun = data[i * 3 + 2]
            if abun != '*':
                NATURAL_ABUNDANCE[number][isotope] = float(abun) / 100.0


# if __name__ == '__main__':
#    print(NAME_TO_CHARGE)
#    print(ISOTOPE_MASS)
#    print(NATURAL_ABUNDANCE)

