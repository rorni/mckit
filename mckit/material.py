# -*- coding: utf-8 -*-

from .constants import NAME_TO_CHARGE, NATURAL_ABUNDANCE, \
                       ISOTOPE_MASS


class Material:
    """Represents material.

    wgt and atomic are both dictionaries: isotope_name -> fraction. If only
    wgt or atomic (and is treated as atomic fraction) present then they not
    need to be normalized.

    Parameters
    ----------
    atomic : dict
        Atomic fractions or concentrations. If none of density and concentration
        as well as wgt present, then atomic is treated as atomic concentrations
        of isotopes. Otherwise it can be only atomic fractions.
    wgt : dict
        Weight fractions of isotopes. density of concentration must present.
    density : float
        Density of the material (g/cc). It is incompatible with concentration
        parameter.
    concentration : float
        Sets the atomic concentration (1 / cc). It is incompatible with density
        parameter.

    Methods
    -------
    density()
        Gets weight density of the material.
    concentration()
        Gets atomic concentration of the material.
    merge(other)
        Merges with other material and returns the result.
    correct(old_vol, new_vol)
        Correct material density - returns the corrected version of the
        material.
    expand()
        Expands elements of natural composition.
    """
    def __init__(self, atomic=None, wgt=None, density=None, concentration=None):
        pass

    def density(self):
        raise NotImplementedError

    def concentration(self):
        raise NotImplementedError

    def merge(self, other):
        raise NotImplementedError

    def correct(self, old_vol, new_vol):
        raise NotImplementedError

    def expand(self):
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

