# -*- coding: utf-8 -*-

from .constants import CHARGE_TO_NAME, NAME_TO_CHARGE, NATURAL_ABUNDANCE, \
                       ISOTOPE_MASS, AVOGADRO


class Material:
    """Represents material.

    Parameters
    ----------
    con_fractions : dict
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
    def __init__(self, con_fractions=None, wgt_fractions=None,
                 density=None, concentration=None):
        pass

    def density2concentration(self, density):
        raise NotImplementedError

    def concentration2density(self, concentration):
        raise NotImplementedError


class Element:
    """Represents isotope or isotope mixture for natural abundance case.

    Parameters
    ----------
    name : str
        Name of isotope. It can be ZAID = Z * 1000 + A, where Z - charge,
        A - the number of protons and neutrons. If A = 0, then natural abundance
        is used. Also it can be an atom_name optionally followed by '-' and A.
        '-' can be omitted. If there is no A, then A is assumed to be 0.
    lib : str, optional
        Name of library.

    Methods
    -------
    charge()
        Gets isotope's electric charge
    expand()
        Expands natural composition of this element.
    mass_number()
        Gets isotope's mass number
    molar_mass()
        Gets isotope's molar mass.
    """
    def __init__(self, name, lib=None):
        z, a = self._split_name(name.upper())
        if z.isalpha():
            self._charge = NAME_TO_CHARGE[z]
        else:
            self._charge = int(z)
        self._mass_number = int(a)
        self._lib = lib

    def __hash__(self):
        return self._charge * self._mass_number * hash(self._lib)

    def __eq__(self, other):
        if self._charge == other.charge() and self._mass_number == \
                other.mass_number() and self._lib == other._lib:
            return True
        else:
            return False

    def charge(self):
        """Gets element's charge number."""
        return self._charge

    def expand(self):
        """Expands natural element into individual isotopes.

        Returns
        -------
        elements : dict
            A dictionary of elements that are comprised by this one.
            Keys - elements - Element instances, values - atomic fractions.
        """
        result = {}
        if self._mass_number > 0:
            result[self] = 1.0
        else:
            for at_num, frac in NATURAL_ABUNDANCE[self._charge].items():
                elem_name = '{0:d}{1:03d}'.format(self._charge, at_num)
                result[Element(elem_name)] = frac
        return result

    def mass_number(self):
        """Gets element's mass number."""
        return self._mass_number

    def molar_mass(self):
        """Gets element's molar mass."""
        Z = self._charge
        A = self._mass_number
        if A > 0:
            if A in ISOTOPE_MASS[Z].keys():
                return ISOTOPE_MASS[Z][A]
            else:   # If no data about molar mass present, then mass number
                return A     # itself is the best approximation.
        else:    # natural abundance
            mol_mass = 0.0
            for at_num, frac in NATURAL_ABUNDANCE[Z].items():
                mol_mass += ISOTOPE_MASS[Z][at_num] * frac
            return mol_mass

    @staticmethod
    def _split_name(name):
        """Splits element name into charge and mass number parts."""
        if name.isnumeric():
            return name[:-3], name[-3:]
        for i, l in enumerate(name):
            if l.isdigit():
                break
        else:
            return name, '0'
        q = name[:i-1] if name[i-1] == '-' else name[:i]
        return q, name[i:]

