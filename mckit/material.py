# -*- coding: utf-8 -*-

import numpy as np

from .constants import NAME_TO_CHARGE, NATURAL_ABUNDANCE, \
                       ISOTOPE_MASS, AVOGADRO


class Material:
    """Represents material.

    wgt and atomic are both lists of tuples: (element, fraction, ...).
    If only wgt or atomic (and is treated as atomic fraction) present then they
    not need to be normalized. atomic's element can be Element instance if
    only atomic parameter present.

    Parameters
    ----------
    atomic : list
        Atomic fractions or concentrations. If none of density and concentration
        as well as wgt present, then atomic is treated as atomic concentrations
        of isotopes. Otherwise it can be only atomic fractions.
    wgt : list
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
    correct(old_vol, new_vol)
        Correct material density - returns the corrected version of the
        material.
    expand()
        Expands elements of natural composition.
    molar_mass()
        Gets molar mass of the material.
    """
    def __init__(self, atomic=[], wgt=[], density=None, concentration=None):
        # Attributes: _n - atomic density (concentration)
        #             _mu - effective molar mass
        self._composition = {}
        elem_w = [Element(item[0]) if not isinstance(item[0], Element)
                  else item[0] for item in wgt]
        elem_a = [Element(item[0]) if not isinstance(item[0], Element)
                  else item[0] for item in atomic]
        frac_w = np.array([item[1] for item in wgt])
        frac_a = np.array([item[1] for item in atomic])
        if density and concentration:
            raise ValueError('density and concentration both must not present.')
        elif atomic and not wgt and not density and not concentration:
            s = np.sum([f * e.molar_mass() for e, f in zip(elem_a, frac_a)])
            for e, v in zip(elem_a, frac_a):
                if e not in self._composition.keys():
                    self._composition[e] = 0.0
                self._composition[e] += v
            self._n = np.sum(frac_a)
            self._mu = s / self._n
        elif (bool(density) ^ bool(concentration)) and \
                (frac_w.size + frac_a.size > 0):
            I_w = np.sum(frac_w)
            I_a = np.sum(frac_a)
            J_w = np.sum(np.divide(frac_w, [e.molar_mass() for e in elem_w]))
            J_a = np.sum(np.multiply(frac_a, [e.molar_mass() for e in elem_a]))

            II_diff = I_a - I_w
            sq_root = np.sqrt(II_diff**2 + 4 * J_w * J_a)
            if II_diff <= 0:
                self._mu = 0.5 * (sq_root - II_diff) / J_w
            else:
                self._mu = 2 * J_a / (sq_root + II_diff)

            norm_factor = self._mu * J_w + I_a
            if concentration:
                self._n = concentration
            else:
                self._n = density * AVOGADRO / self._mu
            coeff = self._mu * self._n / norm_factor
            for el, frac in zip(elem_w, frac_w):
                if el not in self._composition.keys():
                    self._composition[el] = 0.0
                self._composition[el] += coeff * frac / el.molar_mass()
            for el, frac in zip(elem_a, frac_a):
                if el not in self._composition.keys():
                    self._composition[el] = 0.0
                self._composition[el] += self._n / norm_factor * frac
        else:
            raise ValueError('Incorrect set of parameters.')

    def density(self):
        """Gets material's density [g per cc]."""
        return self._n * self._mu / AVOGADRO

    def concentration(self):
        """Gets material's concentration [atoms per cc]."""
        return self._n

    def correct(self, old_vol, new_vol):
        """Creates new material with fixed density to keep cell's mass.
        
        Parameters
        ----------
        old_vol : float
            Initial volume of the cell.
        new_vol : float
            New volume of the cell.
            
        Returns
        -------
        new_mat : Material
            New material that takes into account new volume of the cell.
        """
        factor = old_vol / new_vol
        elements = [(k, v * factor) for k, v in self._composition.items()]
        return Material(atomic=elements)

    def expand(self):
        """Expands natural elements into detailed isotope composition.
        
        Returns
        -------
        new_mat : Material
            New material with detailed isotope composition.
        """
        composition = []
        for el, conc in self._composition.items():
            for isotope, frac in el.expand().items():
                composition.append((isotope, conc * frac))
        return Material(atomic=composition)

    def molar_mass(self):
        """Gets material's effective molar mass [g / mol]."""
        return self._mu


def merge_materials(material1, volume1, material2, volume2):
    """Merges materials.

    Parameters
    ----------
    material1, material2 : Material
        Materials to be merged.
    volume1, volume2 : float
        Volumes of merging cells.
    """
    total_vol = volume1 + volume2
    composition = []
    for el, frac in material1._composition.items():
        composition.append((el, frac * volume1 / total_vol))
    for el, frac in material2._composition.items():
        composition.append((el, frac * volume2 / total_vol))
    return Material(atomic=composition)


def make_mixture(*materials, fraction_type='weight'):
    """Creates new material as a mixture of others.
    
    Fractions are not needed to be normalized, but normalization has effect.
    If the sum of fractions is less than 1, then missing fraction is considered
    to be void (density is reduced). If the sum of fractions is greater than 1,
    the effect of compression is taking place.
    
    Parameters
    ----------
    materials : list
        A list of pairs material-fraction. material must be an Material class
        instance because for mixture not only composition but density is 
        important.
    fraction_type : str
        Indicate how fraction should be interpreted.
        'weight' - weight fractions (default);
        'volume' - volume fractions;
        'atomic' - atomic fractions.

    Returns
    -------
    material : Material
        New material.
    """
    elements = {}
    if fraction_type == 'weight':
        s = np.sum([frac / (mat._mu * mat._n) for mat, frac in materials])
        norm = lambda m: 1.0 / (m._mu * s)
    elif fraction_type == 'volume':
        norm = lambda m: m._n
    elif fraction_type == 'atomic':
        s = np.sum([frac / mat._n for mat, frac in materials])
        norm = lambda m: 1.0 / s
    else:
        raise ValueError('Unknown fraction type')
    for mat, frac in materials:
        for el, conc in mat._composition:
            if el not in elements.keys():
                elements[el] = 0.0
            elements[el] += frac * conc * norm(mat)
    return Material(atomic=elements.items())


class Element:
    """Represents isotope or isotope mixture for natural abundance case.

    Parameters
    ----------
    name : str or int
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
        if isinstance(name, int):
            self._charge = name // 1000
            self._mass_number = name % 1000
        else:
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

