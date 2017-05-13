# -*- coding: utf-8 -*-

from .constants import ATOM_NAMES, NATURAL_ABUNDANCE, ISOTOPE_MASS


class Material:
    """Represents material.

    Parameters
    ----------
    composition : dict
        Sets composition of the material. Keys - isotope names, values - weight
        or atomic concentration.

    Methods
    -------
    density2concentration(density)
        Converts weight density to concentration of atoms.
    concentration2density(concentration)
        Converts concentration of atoms to weight density.
    """
    def __init__(self, composition):
        pass

    def density2concentration(self, density):
        raise NotImplementedError

    def concentration2density(self, concentration):
        raise NotImplementedError


def molar_mass(isotope_name):
    """Gets isotope molar mass.

    Parameters
    ----------
    isotope_name : int or str
        Name of isotope. If int - ZAID = Z * 1000 + A, where Z - charge,
        A - the number of protons and neutrons. If A = 0, then natural abundance
        is used. If str then it is atom_name + '-' + A. '-' + A is optional - in
        this case A assumed to be equal to zero (see above).

    Returns
    -------
    mol_mass : float
        Molar mass [g / mol].
    """
    if isinstance(isotope_name, int):
        z = isotope_name // 1000
        a = isotope_name % 1000
    else:
        comp = isotope_name.split('-')
        try:
            z = ATOM_NAMES[comp[0].upper()]
        except KeyError:
            raise ValueError('Unknown atom name: {0}'.format(comp[0]))
        a = 0 if len(comp) == 1 else int(comp[1])

    if a > 0:       # Individual isotope
        if a in ISOTOPE_MASS[z].keys():
            return ISOTOPE_MASS[z][a]
        else:           # If no data about molar mass present then A itself is
            return a    # the best approximation.
    else:           # Natural abundance
        mol_mass = 0.0
        for at_num, frac in NATURAL_ABUNDANCE[z].items():
            mol_mass += ISOTOPE_MASS[z][at_num] * frac
        return mol_mass

