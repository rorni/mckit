# -*- coding: utf-8 -*-

import math
from functools import reduce
from operator import xor

import numpy as np

from .constants import NAME_TO_CHARGE, NATURAL_ABUNDANCE, \
                       ISOTOPE_MASS, AVOGADRO, \
                       RELATIVE_COMPOSITION_TOLERANCE,\
                       RELATIVE_DENSITY_TOLERANCE, CHARGE_TO_NAME
from .printer import print_card, MCNP_FORMATS


class Composition:
    """Represents composition.

    Composition is not a material. It specifies only isotopes and their
    fractions. It doesn't concern absolute quantities like density and
    concentration. Composition immediately corresponds to the MCNP's
    material.

    weight and atomic are both lists of tuples: (element, fraction, ...).

    Parameters
    ----------
    atomic : list
        Atomic fractions.
    weight : list
        Weight fractions.
    options : dict
        Dictionary of composition options.

    Methods
    -------
    molar_mass()
        Gets molar mass of the composition.
    get_atomic(isotope)
        Gets atomic fraction of the isotope.
    get_weight(isotope)
        Gets weight fraction of the isotope.
    expand()
        Expands natural abundance elements into detailed isotope composition.
    natural(tolerance)
        Try to replace detailed isotope composition with natural elements.
    """
    def __init__(self, atomic=(), weight=(), **options):
        self._composition = {}
        elem_w = []
        frac_w = []
        for elem, frac in weight:
            if isinstance(elem, Element):
                elem_w.append(elem)
            else:
                elem_w.append(Element(elem))
            frac_w.append(frac)

        elem_a = []
        frac_a = []
        for elem, frac in atomic:
            if isinstance(elem, Element):
                elem_a.append(elem)
            else:
                elem_a.append(Element(elem))
            frac_a.append(frac)

        if len(frac_w) + len(frac_a) > 0:
            I_w = np.sum(frac_w)
            I_a = np.sum(frac_a)
            J_w = np.sum(np.divide(frac_w, [e.molar_mass for e in elem_w]))
            J_a = np.sum(np.multiply(frac_a, [e.molar_mass for e in elem_a]))

            II_diff = I_a - I_w
            sq_root = np.sqrt(II_diff ** 2 + 4 * J_w * J_a)
            if II_diff <= 0:
                self._mu = 0.5 * (sq_root - II_diff) / J_w
            else:
                self._mu = 2 * J_a / (sq_root + II_diff)

            norm_factor = self._mu * J_w + I_a
            for el, frac in zip(elem_w, frac_w):
                if el not in self._composition.keys():
                    self._composition[el] = 0.0
                self._composition[el] += self._mu / norm_factor * frac / el.molar_mass
            for el, frac in zip(elem_a, frac_a):
                if el not in self._composition.keys():
                    self._composition[el] = 0.0
                self._composition[el] += frac / norm_factor
        else:
            raise ValueError('Incorrect set of parameters.')
        self._expanded = False
        self.expand()
        self._hash = reduce(xor, map(hash, self._composition.keys()))
        self._options = options.copy()

    def __eq__(self, other):
        if not self._expanded:
            self.expand()
        if not other._expanded:
            other.expand()
        if len(self._composition.keys()) != len(other._composition.keys()):
            return False
        for k1, v1 in self._composition.items():
            v2 = other._composition.get(k1, None)
            if v2 is None:
                return False
            if not math.isclose(v1, v2, rel_tol=RELATIVE_COMPOSITION_TOLERANCE):
                return False
        return True

    def __hash__(self):
        return self._hash

    def __str__(self):
        text = ['M' + str(self['name']), ' ']
        for elem, frac in self._composition.items():
            text.append(elem.mcnp_repr())
            text.append('  ')
            text.append(MCNP_FORMATS['material_fraction'].format(frac))
            text.append('\n')
        return print_card(text)

    def __getitem__(self, key):
        return self._options[key]

    def __iter__(self):
        return iter(self._composition.items())

    def __contains__(self, item):
        """Checks if the composition contains the item.

        Parameters
        ----------
        item : str or Element
            Isotope. It can be either isotope name or Element instance.

        Returns
        -------
        result : bool
            True if the composition contains the isotope, False otherwise.
        """
        if not isinstance(item, Element):
            item = Element(item)
        return item in self._composition

    def get_atomic(self, isotope):
        """Gets atomic fraction of the isotope.

        Raises KeyError if the composition doesn't contain the isotope.

        Parameters
        ----------
        isotope : str or Element
            Isotope. It can be either isotope name or Element instance.

        Returns
        -------
        frac : float
            Atomic fraction of the specified isotope.
        """
        if not isinstance(isotope, Element):
            isotope = Element(isotope)
        return self._composition[isotope]

    def get_weight(self, isotope):
        """Gets weight fraction of the isotope.

        Raises KeyError if the composition doesn't contain the isotope.

        Parameters
        ----------
        isotope : str or Element
            Isotope. It can be either isotope name or Element instance.

        Returns
        -------
        frac : float
            Weight fraction of the specified isotope.
        """
        if not isinstance(isotope, Element):
            isotope = Element(isotope)
        at = self._composition[isotope]
        return at * isotope.molar_mass / self._mu

    @property
    def molar_mass(self):
        """Gets composition's effective molar mass [g / mol]."""
        return self._mu

    def expand(self):
        """Expands elements with natural abundances into detailed isotope composition.

        Modifies current object.
        """
        self._expanded = True
        composition = {}
        for el, conc in self._composition.items():
            for isotope, frac in el.expand().items():
                if isotope not in composition.keys():
                    composition[isotope] = 0
                composition[isotope] += conc * frac
        self._composition = composition

    def natural(self, tolerance=1.e-8):
        """Tries to replace detailed isotope composition by natural elements.

        Modifies current object.

        Parameters
        ----------
        tolerance : float
            Relative tolerance to consider isotope fractions as equal.
            Default: 1.e-8

        Returns
        -------
        comp : Composition
            self, if composition is reduced successfully to natural. None returned if the
            composition cannot be reduced to natural.
        """
        already = True
        by_charge = {}
        for elem, frac in self._composition.items():
            q = elem.charge
            if q not in by_charge.keys():
                by_charge[q] = {}
            a = elem.mass_number
            if a > 0:
                already = False
            if a not in by_charge[q].keys():
                by_charge[q][a] = 0
            by_charge[q][a] += frac

        if already:  # No need for further checking - only natural elements present.
            return self

        composition = {}
        for q, isotopes in by_charge.items():
            frac_0 = isotopes.pop(0, None)
            tot_frac = sum(isotopes.values())
            for a, frac in isotopes.items():
                ifrac = frac / tot_frac
                delta = 2 * abs(ifrac - NATURAL_ABUNDANCE[q][a]) / (ifrac + NATURAL_ABUNDANCE[q][a])
                if delta > tolerance:
                    return None
            elem = Element(q * 1000)
            composition[elem] = tot_frac
            if frac_0:
                composition[elem] += frac_0
        self._composition = composition
        self._expanded = False
        return self

    def elements(self):
        """Gets iterator over composition's elements."""
        return iter(self._composition.keys())

    @staticmethod
    def mixture(*compositions):
        """Makes mixture of the compositions with specific fractions.

        Parameters
        ----------
        compositions : list
            List of pairs composition, fraction.

        Returns
        -------
        mix : Composition
            Mixture.
        """
        atomics = []
        if len(compositions) == 1:
            return compositions[0][0]
        for comp, frac in compositions:
            for elem, atom_frac in comp:
                atomics.append((elem, atom_frac * frac))
        return Composition(atomic=atomics)


class Material:
    """Represents material.

    wgt and atomic are both lists of tuples: (element, fraction, ...).
    If only wgt or atomic (and is treated as atomic fraction) present then they
    not need to be normalized. atomic's element can be Element instance if
    only atomic parameter present.

    Parameters
    ----------
    atomic : list
        Atomic fractions. New composition will be created.
    weight : list
        Weight fractions of isotopes. density of concentration must present.
    composition : Composition
        Composition instance. If it is specified, then this composition will be
        used. Neither atomic nor weight must be present.
    density : float
        Density of the material (g/cc). It is incompatible with concentration
        parameter.
    concentration : float
        Sets the atomic concentration (1 / cc). It is incompatible with density
        parameter.
    options : dict
        Extra options.

    Properties
    ----------
    density : float
        Density of the material [g/cc].
    concentration : float
        Concentration of the material [atoms/cc].
    composition : Composition
        Material's composition.
    molar_mass : float
        Material's molar mass [g/mol].

    Methods
    -------
    correct(old_vol, new_vol)
        Correct material density - returns the corrected version of the
        material.
    mixture(*materials, fraction_type)
        Makes a mixture of specified materials in desired proportions.
    __getitem__(key)
        Gets specific option.
    """
    def __init__(self, atomic=(), weight=(), composition=None, density=None, concentration=None, **options):
        # Attributes: _n - atomic density (concentration)
        if isinstance(composition, Composition) and not atomic and not weight:
            self._composition = composition
        elif not composition and (weight or atomic):
            self._composition = Composition(atomic=atomic, weight=weight, **options)
        else:
            raise ValueError("Incorrect set of parameters.")

        if concentration and density or not concentration and not density:
            raise ValueError("Incorrect set of parameters.")
        elif concentration:
            self._n = concentration
        else:
            self._n = density * AVOGADRO / self._composition.molar_mass
        self._options = options

    def __eq__(self, other):
        if not math.isclose(self._n, other.concentration, rel_tol=RELATIVE_DENSITY_TOLERANCE):
            return False
        return self._composition == other.composition

    def __hash__(self):
        return hash(self._composition)

    def __getitem__(self, key):
        return self._options[key]

    @property
    def density(self):
        """Gets material's density [g per cc]."""
        return self._n * self._composition.molar_mass / AVOGADRO

    @property
    def concentration(self):
        """Gets material's concentration [atoms per cc]."""
        return self._n

    @property
    def composition(self):
        """Gets Composition instance that corresponds to the material."""
        return self._composition

    @property
    def molar_mass(self):
        """Gets material's effective molar mass [g / mol]."""
        return self._composition.molar_mass

    def correct(self, old_vol=None, new_vol=None, factor=None):
        """Creates new material with fixed density to keep cell's mass.

        Either old_vol and new_vol or factor must be specified.

        Parameters
        ----------
        old_vol : float
            Initial volume of the cell.
        new_vol : float
            New volume of the cell.
        factor : float
            By this factor density of material will be multiplied. If factor
            is specified, then its value will be used.
            
        Returns
        -------
        new_mat : Material
            New material that takes into account new volume of the cell.
        """
        if factor is None:
            factor = old_vol / new_vol
        return Material(composition=self._composition, concentration=self._n * factor, **self._options)

    @staticmethod
    def mixture(*materials, fraction_type='weight'):
        """Creates new material as a mixture of others.

        Volume fractions are not needed to be normalized, but normalization has effect.
        If the sum of fractions is less than 1, then missing fraction is considered
        to be void (density is reduced). If the sum of fractions is greater than 1,
        the effect of compression is taking place. But for weight and atomic fractions
        normalization will be done anyway.

        Parameters
        ----------
        materials : list
            A list of pairs material-fraction. material must be a Material class
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
        if not materials:
            raise ValueError('At least one material must be specified.')
        if fraction_type == 'weight':
            fun = lambda m, f: f / m.molar_mass
            norm = sum(fun(m, f) / m.concentration for m, f in materials)
        elif fraction_type == 'volume':
            fun = lambda m, f: f * m.concentration
            norm = 1
        elif fraction_type == 'atomic':
            fun = lambda m, f: f
            norm = sum(fun(m, f) / m.concentration for m, f in materials)
        else:
            raise ValueError('Unknown fraction type')
        factor = sum([fun(m, f) for m, f in materials])
        compositions = [(m.composition, fun(m, f) / factor) for m, f in materials]
        new_comp = Composition.mixture(*compositions)
        return Material(composition=new_comp, concentration=factor / norm)


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
    isomer : int
        Isomer level. Default: 0 - ground state.
    comment : str, optional
        Optional comment to the element.

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
    def __init__(self, name, lib=None, isomer=0, comment=None):
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

        # molar mass calculation
        Z = self._charge
        A = self._mass_number
        if A > 0:
            if A in ISOTOPE_MASS[Z].keys():
                self._molar = ISOTOPE_MASS[Z][A]
            else:  # If no data about molar mass present, then mass number
                self._molar = A  # itself is the best approximation.
        else:  # natural abundance
            self._molar = 0.0
            for at_num, frac in NATURAL_ABUNDANCE[Z].items():
                self._molar += ISOTOPE_MASS[Z][at_num] * frac
        # Other flags and parameters
        self._lib = lib
        if self._mass_number == 0:
            isomer = 0
        self._isomer = isomer
        self._comment = comment

    def __hash__(self):
        return self._charge * self._mass_number * hash(self._lib) * (self._isomer + 1)

    def __eq__(self, other):
        if self._charge == other.charge and self._mass_number == \
                other.mass_number and self._lib == other._lib and self._isomer == other._isomer:
            return True
        else:
            return False

    def __str__(self):
        name = CHARGE_TO_NAME[self.charge].capitalize()
        if self._mass_number > 0:
            name += str(self._mass_number)
            if self._isomer > 0:
                name += 'm'
            if self._isomer > 1:
                name += str(self._isomer - 1)
        return name

    def mcnp_repr(self):
        """Returns MCNP representation of the element."""
        name = str(self.charge * 1000 + self.mass_number)
        if self.lib is not None:
            name += '.{0}'.format(self.lib)
        return name

    @property
    def charge(self):
        """Gets element's charge number."""
        return self._charge

    @property
    def mass_number(self):
        """Gets element's mass number."""
        return self._mass_number

    @property
    def molar_mass(self):
        """Gets element's molar mass."""
        return self._molar

    @property
    def lib(self):
        """Gets library name."""
        return self._lib

    @property
    def isomer(self):
        """Gets isomer level."""
        return self._isomer

    def expand(self):
        """Expands natural element into individual isotopes.

        Returns
        -------
        elements : dict
            A dictionary of elements that are comprised by this one.
            Keys - elements - Element instances, values - atomic fractions.
        """
        result = {}
        if self._mass_number > 0 and self._mass_number in NATURAL_ABUNDANCE[self._charge].keys():
            result[self] = 1.0
        elif self._mass_number == 0:
            for at_num, frac in NATURAL_ABUNDANCE[self._charge].items():
                elem_name = '{0:d}{1:03d}'.format(self._charge, at_num)
                result[Element(elem_name, lib=self._lib)] = frac
        return result

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

