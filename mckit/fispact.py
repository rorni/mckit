"""Module for activation calculations and FISPACT coupling."""
from itertools import accumulate
from abc import ABC, abstractmethod
from functools import reduce
from collections import defaultdict
import random
import re
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .constants import TIME_UNITS
from .parser.fispact_parser import read_fispact_tab
from .fmesh import SparseData
from .body import Body


__all__ = ['activation', 'mesh_activation']


EBINS_24 = [
    0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.22,
    1.44, 1.66, 2.00, 2.50, 3.00, 4.00, 5.00, 6.50, 8.00, 10.0, 12.0, 14.0, 20.0
]

EBINS_22 = [
    0.00, 0.01, 0.10, 0.20, 0.40, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
    5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 10.0, 12.0, 14.0
]

DATA_PATH = r'D:\\nuclear_data\\fispact\\ENDFdata\\'

LIBS = {
    'ind_nuc':  r'TENDL2014data\\tendl14_decay12_index',
    'xs_endf':  r'TENDL2014data\\tal2014-n\\gxs-709',
    'xs_endfb': r'TENDL2014data\\tal2014-n\\tal2014-n.bin',
    'prob_tab': r'TENDL2014data\\tal2014-n\\tp-709-294',
    'dk_endf':  r'decay\\decay_2012',
    'fy_endf':  r'TENDL2014data\\tal2014-n\\gef42_nfy',
    'sf_endf':  r'TENDL2014data\\tal2014-n\\gef42_sfy',
    'hazards':  r'decay\\hazards_2012',
    'clear':    r'decay\\clear_2012',
    'a2data':   r'decay\\a2_2012',
    'absorp':   r'decay\\abs_2012'
}


class FispactError(Exception):
    pass


class FispactSettings:
    """FISPACT settings object.

    Parameters
    ----------
    irr_profile : IrradiationProfile
        Irradiation profile.
    relax_profile : IrradiationProfile
        Relaxation profile.
    nat_reltol : float
        Relative tolerance to believe that elements have natural abundance.
        To force use of isotopic composition set nat_reltol to None.
        Default: 1.e-8.
    use_binary : bool
        Use binary libraries rather then text. Default: False.
    zero : bool
        If True, then time value is reset to zero after an irradiation.
    mind : float
        Indicate the minimum number of atoms which are regarded as significant
        for the output inventory. Default: 1.e+5
    use_fission : bool
        Causes to use fission reactions. If it is False - fission reactions are
        omitted. Default: False - not to use fission.
    half : bool
        If True, causes the half-lije of each nuclide to be printed in the
        output at all timesteps. Default: True.
    hazards : bool
        If True, causes data on potential ingestion and inhalation doses to be
        read and dose due to individual nuclides to be printed at all timesteps.
        Default: False.
    tab1, tab2, tab3, tab4: bool
        If True, causes output of the specific data into separate files.
        tab1 - number of atoms and grams of each nuclide;
        tab2 - activity (Bq) and dose rate (Sv per hour) of each nuclide;
        tab3 - ingestion and inhalation dose (Sv) of each nuclide;
        tab4 - gamma-ray spectrum (MeV per sec) and the number of gammas per group.
        All defaults to False.
    nostable : bool
        If True, printing of stable nuclides in the inventory is suppressed.
        Default: False
    inv_tol : (float, float)
        (atol, rtol) - absolute and relative tolerances for inventory
        calculations. Default: None - means default values remain (1.e+4, 2.e-3).
    path_tol : (float, float)
        (atol, rtol) - absolute and relative tolerances for pathways calculations.
        Default: None - means default values remain (1.e+4, 2.e-3).
    uncertainty : int
        Controls the uncertainty estimates and pathway information that are
        calculated and output for each time interval. Default: 0.
        0 - no pathways or estimates of uncertainty are calculated or output;
        1 - only estimates of uncertainty are output;
        2 - both estimates of uncertainty and the pathway information are output;
        3 - only the pathway information is output;
    """
    def __init__(self, irr_profile, nat_reltol=1.e-8, use_binary=False,
                 zero=True, mind=1.e+5, use_fission=False, half=True,
                 hazards=False, tab1=False, tab2=False, tab3=False, tab4=False,
                 nostable=False, inv_tol=None, path_tol=None, uncertainty=0):
        self.irr_profile = irr_profile
        self.nat_reltol = nat_reltol
        self.use_binary = use_binary
        self._kwargs = {
            'ZERO': zero, 'MIND': mind, 'USEFISSION': use_fission, 'HALF': half,
            'HAZARDS': hazards, 'TAB1': tab1, 'TAB2': tab2, 'TAB3': tab3,
            'TAB4': tab4, 'NOSTABLE': nostable, 'INVENTORY_TOLERANCE': inv_tol,
            'PATH_TOLERANCE': path_tol, 'UNCERTAINTY': uncertainty
        }

    def get_param(self, name):
        """Gets FISPACT inventory parameter.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        value : object
            Parameter's value.
        """
        return self._kwargs.get(name)


def run_fispact(input_file, files='files', cwd=None, verbose=True):
    """Runs FISPACT code.

    If run ends with errors, then FispactError exception is raised.

    Parameters
    ----------
    input_file : str
        The name of input file.
    files : str
        The name of FISPACT files file. Default: 'files'.
    cwd : Path-like or str
        Working directory. Default: None.
    verbose : bool
        Whether to print calculation status to stdout.

    Returns
    -------
    status : str
        Run status message.
    """
    status = subprocess.check_output(
        ['fispact', input_file, files], encoding='utf-8', cwd=cwd
    )
    if verbose:
        print(status)
    check_fispact_status(status)
    return status


def check_fispact_status(text):
    """Raises FispactError exception if FATAL ERROR presents in output.

    Parameters
    ----------
    text : str
        Text to be checked.
    """
    match = re.search('^.*run +terminated.*$', text, flags=re.MULTILINE)
    if match:
        raise FispactError(match.group(0))


def create_files(files='files', collapx='COLLAPX', fluxes='fluxes',
                 arrayx='ARRAYX', cwd=Path()):
    """Creates new files file, that specifies fispact names and data.

    Parameters
    ----------
    files : str
        Name of file with list of libraries and other useful files. Default: files
    collapx : str
        Name of file of the collapsed cross sections. Default: COLLAPX
    fluxes : str
        Name of file with flux data. Default: fluxes.
    arrayx : str
        Name of arrayx file. Usually it is needed to be calculated only once.
    cwd : Path
        Working directory. In this directory files will be created. The folder
        must be exist. Default: current directory.

    Returns
    -------
    files : str
        Name of files file.
    """
    with open(cwd / files, mode='w') as f:
        for k, v in LIBS.items():
            f.write(k + '  ' + DATA_PATH + v + '\n')
        f.write('fluxes  ' + fluxes + '\n')
        f.write('collapxi  ' + collapx + '\n')
        f.write('collapxo  ' + collapx + '\n')
        f.write('arrayx  ' + arrayx + '\n')
    return files


def create_convert(ebins, flux, convert='convert', fluxes='fluxes',
                   arb_flux='arb_flux', files='files.convert', cwd=Path()):
    """Creates file for fispact flux conversion to the 709 groups.

    Parameters
    ----------
    ebins : array_like[float]
        Energy bins.
    flux : array_like[float]
        Group flux.
    convert : str
        File name for convert input file.
    fluxes : str
        File name for converted neutron flux.
    arb_flux : str
        File name for input neutron flux.
    files : str
        File name for conversion data.
    cwd : Path
        Working directory. In this directory files will be created. The folder
        must be exist.

    Returns
    -------
    args : tuple
        Tuple of arguments for FISPACT.
    """
    with open(cwd / files, mode='w') as f:
        f.write('ind_nuc  ' + DATA_PATH + LIBS['ind_nuc'] + '\n')
        f.write('fluxes  ' + fluxes + '\n')
        f.write('arb_flux  ' + arb_flux + '\n')

    with open(cwd / arb_flux, mode='w') as f:
        ncols = 6
        text = []
        for i, e in enumerate(reversed(ebins)):
            s = '\n' if (i + 1) % ncols == 0 else ' '
            text.append('{0:.6e}'.format(e * 1.e+6))  # Because fispact needs
            text.append(s)                            # eV, not MeV
        text[-1] = '\n'
        f.write(''.join(text))

        text = []
        for i, e in enumerate(reversed(flux)):
            s = '\n' if (i + 1) % ncols == 0 else ' '
            text.append('{0:.6e}'.format(e))
            text.append(s)
        text[-1] = '\n'
        f.write(''.join(text))
        f.write('{0}\n'.format(1))
        f.write('total flux={0:.6e}'.format(np.sum(flux)))

    with open(str(cwd / convert) + '.i', mode='w') as f:
        text = [
            '<< convert flux to 709 grout structure >>',
            'CLOBBER',
            'GRPCONVERT {0} 709'.format(len(flux)),
            'FISPACT',
            '* SPECTRAL MODIFICATION',
            'END',
            '* END'
        ]
        f.write('\n'.join(text))
    return convert, files


def create_collapse(collapse='collapse', use_binary=True, cwd=Path()):
    """Creates fispact file for cross-section collapse.

    Parameters
    ----------
    collapse : str
        Filename for collapse input file.
    use_binary : bool
        Use binary data rather text data.
    cwd : Path
        Working directory. In this directory files will be created.
        Default: current directory.

    Returns
    -------
    collapse : str
        Name of collapse file.
    """
    p = -1 if use_binary else +1
    with open(str(cwd / collapse) + '.i', mode='w') as f:
        text = [
            '<< collapse cross section data >>',
            'CLOBBER',
            'GETXS {0} 709'.format(p),
            'FISPACT',
            '* COLLAPSE',
            'END',
            '* END OF RUN'
        ]
        f.write('\n'.join(text))
    return collapse


def create_condense(condense='condense', cwd=Path()):
    """Creates fispact file to condense the decay and fission data.

    Parameters
    ----------
    condense : str
        Name of condense input file.
    cwd : Path
        Working directory. In this directory files will be created. Default:
        current directory.

    Returns
    -------
    condense : str
        Name of condense file.
    """
    with open(str(cwd / condense) + '.i', mode='w') as f:
        text = [
            '<< Condense decay data >>',
            'CLOBBER',
            'SPEK',
            'GETDECAY 1',
            'FISPACT',
            '* CONDENSE',
            'END',
            '* END OF RUN'
        ]
        f.write('\n'.join(text))
    return condense


def create_inventory(title, material, volume, flux, inventory='inventory',
                     settings=None, cwd=Path()):
    """Creates fispact file for inventory calculations.

    Parameters
    ----------
    title : str
        Title for the inventory.
    material : Material
        Material to be irradiated.
    volume : float
        Volume of the material.
    flux : float
        Total neutron flux.
    inventory : str
        File name for inventory input file.
    settings : FispactSettings
        Object that represents FISPACT inventory calculations parameters.
    cwd : Path
        Working directory. In this directory files will be created.
        Default: current directory.

    Returns
    -------
    inventory : str
        Name of inventory file.
    """
    if settings is None:
        settings = FispactSettings()
    # inventory header.
    text = [
        '<< {0} >>'.format(title),
        'CLOBBER',
        'GETXS 0',
        'GETDECAY 0',
        'FISPACT',
        '* {0}'.format(title)
    ]
    # Initial conditions.
    # ------------------
    # Material
    text.extend(print_material(material, volume, tolerance=settings.nat_reltol))
    # Calculation parameters.
    text.append('MIND  {0:.5e}'.format(settings.get_param('MIND')))
    if settings.get_param('USEFISSION'):
        text.append('USEFISSION')
    if settings.get_param('HALF'):
        text.append('HALF')
    if settings.get_param('HAZARDS'):
        text.append('HAZARDS')
    if settings.get_param('TAB1'):
        text.append('TAB1 1')
    if settings.get_param('TAB2'):
        text.append('TAB2 1')
    if settings.get_param('TAB3'):
        text.append('TAB3 1')
    if settings.get_param('TAB4'):
        text.append('TAB4 1')
    if settings.get_param('NOSTABLE'):
        text.append('NOSTABLE')
    inv_tol = settings.get_param('INVENTORY_TOLERANCE')
    if inv_tol:
        text.append('TOLERANCE  0  {0:.5e}  {1:.5e}'.format(*inv_tol))
    path_tol = settings.get_param('PATH_TOLERANCE')
    if path_tol:
        text.append('TOLERANCE  1  {0:.5e}  {1:.5e}'.format(*path_tol))
    uncertainty = settings.get_param('UNCERTAINTY')
    if uncertainty:
        text.append('UNCERTAINTY {0}'.format(uncertainty))
    # Irradiation and relaxation profiles
    text.extend(settings.irr_profile.output(flux))
    # Footer
    text.append('END')
    text.append('* END of calculations')
    # Save to file
    with open(str(cwd / inventory) + '.i', mode='w') as f:
        f.write('\n'.join(text))
    return inventory


def print_material(material, volume, tolerance=1.e-8):
    """Produces FISPACT description of the material.

    Parameters
    ----------
    material : Material
        Material to be irradiated.
    volume : float
        Volume of the material.
    tolerance : float
        Relative tolerance to believe that isotopes have natural abundance.
        If None - no checking is performed and FUEL keyword is used.

    Returns
    -------
    text : list[str]
        List of words.
    """
    text = ['DENSITY {0}'.format(material.density)]
    composition = []
    if tolerance is not None:
        nat_comp = material.composition.natural(tolerance)
        if nat_comp is not None:
            mass = volume * material.density / 1000    # Because mass must be specified in kg.
            for e in nat_comp.elements():
                composition.append((e, nat_comp.get_weight(e) * 100))
            text.append('MASS {0:.5} {1}'.format(mass, len(composition)))
    else:
        nat_comp = None

    if tolerance is None or nat_comp is None:
        exp_comp = material.composition.expand()
        tot_atoms = volume * material.concentration
        # print('tot atoms ', tot_atoms, 'vol ', volume, 'conc ', material.concentration)
        for e in exp_comp.elements():
            composition.append((e, exp_comp.get_atomic(e) * tot_atoms))
        text.append('FUEL  {0}'.format(len(composition)))

    for e, f in sorted(composition, key=lambda x: -x[1]):
        # print(e, f)
        text.append('  {0:2s}   {1:.5e}'.format(e.fispact_repr(), f))
    return text


def irradiation_SA2():
    """Creates SA2 irradiation profile.

    Returns
    -------
    irr_prof : IrradiationProfile
        Irradiation profile object.
    """
    irr_prof = IrradiationProfile(4.5643E+12)
    irr_prof.irradiate(2.4452E+10, 2, units='YEARS', record='SPEC')
    irr_prof.irradiate(1.8828E+11, 10, units='YEARS', record='SPEC')
    irr_prof.relax(0.667, units='YEARS', record='SPEC')
    irr_prof.irradiate(3.7900E+11, 1.330, units='YEARS', record='SPEC')
    for i in range(17):
        irr_prof.relax(3920, record='SPEC')
        irr_prof.irradiate(4.5643E+12, 400, record='SPEC')
    irr_prof.relax(3920, record='SPEC')
    irr_prof.irradiate(6.3900E+12, 400, record='SPEC')
    irr_prof.relax(3920, record='SPEC')
    irr_prof.irradiate(6.3900E+12, 400, record='SPEC')
    irr_prof.relax(3920, record='SPEC')
    irr_prof.irradiate(6.3900E+12, 400, record='SPEC')
    irr_prof.relax(1, record='ATOMS')
    irr_prof.relax(299, record='ATOMS')
    irr_prof.relax(25, units='MINS', record='ATOMS')
    irr_prof.relax(30, units='MINS', record='ATOMS')
    irr_prof.relax(2, units='HOURS', record='ATOMS')
    irr_prof.relax(2, units='HOURS', record='ATOMS')
    irr_prof.relax(5, units='HOURS', record='ATOMS')
    irr_prof.relax(14, units='HOURS', record='ATOMS')
    irr_prof.relax(2, units='DAYS', record='ATOMS')
    irr_prof.relax(4, units='DAYS', record='ATOMS')
    irr_prof.relax(23, units='DAYS', record='ATOMS')
    irr_prof.relax(60, units='DAYS', record='ATOMS')
    irr_prof.relax(275.25, units='DAYS', record='ATOMS')
    irr_prof.relax(2, units='YEARS', record='ATOMS')
    irr_prof.relax(7, units='YEARS', record='ATOMS')
    irr_prof.relax(20, units='YEARS', record='ATOMS')
    irr_prof.relax(20, units='YEARS', record='ATOMS')
    irr_prof.relax(50, units='YEARS', record='ATOMS')
    irr_prof.relax(900, units='YEARS', record='ATOMS')
    return irr_prof


def cooling_SA2():
    """Creates cooling profile for SA2 scenario.

    Returns
    -------
    cool_prof : IrradiationProfile
        Cooling profile object.
    """
    cool_prof = IrradiationProfile()
    cool_prof.relax(1, record='ATOMS')
    cool_prof.relax(299, record='ATOMS')
    cool_prof.relax(25, units='MINS', record='ATOMS')
    cool_prof.relax(30, units='MINS', record='ATOMS')
    cool_prof.relax(2, units='HOURS', record='ATOMS')
    cool_prof.relax(2, units='HOURS', record='ATOMS')
    cool_prof.relax(5, units='HOURS', record='ATOMS')
    cool_prof.relax(14, units='HOURS', record='ATOMS')
    cool_prof.relax(2, units='DAYS', record='ATOMS')
    cool_prof.relax(4, units='DAYS', record='ATOMS')
    cool_prof.relax(23, units='DAYS', record='ATOMS')
    cool_prof.relax(60, units='DAYS', record='ATOMS')
    cool_prof.relax(275.25, units='DAYS', record='ATOMS')
    cool_prof.relax(2, units='YEARS', record='ATOMS')
    cool_prof.relax(7, units='YEARS', record='ATOMS')
    cool_prof.relax(20, units='YEARS', record='ATOMS')
    cool_prof.relax(20, units='YEARS', record='ATOMS')
    cool_prof.relax(50, units='YEARS', record='ATOMS')
    cool_prof.relax(900, units='YEARS', record='ATOMS')
    return cool_prof


class FispactResult(ABC):
    """Represents a result of FISPACT calculations.

    Methods
    -------
    add_frame(data, duration, units)
        Adds new time frame with calculation results.
    """
    def __init__(self):
        self._times = TimeSeries()
        self._data = []

    def add_frame(self, data, duration, units='SECS'):
        """Adds new timeframe with calculation results.

        Parameters
        ----------
        data : object
            Data item.
        duration : float
            Duration of timeframe.
        units : str
            Units, in which duration is measured. Default: 'SECS'.
        """
        self._times.append_interval(duration, units=units)
        self._data.append(data)

    def __getitem__(self, index):
        return self._times[index], self._data[index]

    def __iter__(self):
        return zip(self._times, self._data)

    @abstractmethod
    def as_matrix(self, encoding_rules=None):
        """Gets representation of data as matrix.

        If data is not a vector or float number, encoding_rules represents
        rules which is used to order elements of data.

        Parameters
        ----------
        encoding_rules : dict
            A dictionary of pairs item->number.

        Returns
        -------
        matrix : numpy.ndarray
            Data represented as matrix. First axis is time.
        """


class GammaSpectra(FispactResult):
    """Represents activation gamma source."""
    def as_matrix(self, encoding_rules=None):
        nt = len(self._times)
        ne = len(self._data[0])
        matrix = np.empty((nt, ne), dtype=float)
        for i, flux in enumerate(self._data):
            matrix[i, :] = flux
        return matrix


class IsotopeData(FispactResult):
    """Represents any activation data for individual elements."""
    def as_matrix(self, encoding_rules=None):
        nt = len(self._times)
        ne = len(encoding_rules.values())
        matrix = np.zeros((nt, ne), dtype=float)
        for i, data in enumerate(self._data):
            if data:
                for element, value in data.items():
                    j = encoding_rules.get(element, None)
                    if j:
                        matrix[i, j] = value
        return matrix

    def get_isotopes(self):
        """Gets all isotopes present in the inventory.

        Returns
        -------
        isotopes : set
            Set of isotopes (material.Element)
        """
        isotopes = set()
        for data in self._data:
            if data:
                isotopes.update(data.keys())
        return isotopes


class TimeSeries:
    """Represents an array of time points.

    Methods
    -------
    append_interval(delta, units)
        Appends a time moment after delta pass since last time point.
    insert_point(time, units)
        Inserts a time point.
    durations()
        Gets an iterator over time intervals between time points.
    adjust_time(time)
        Gets time in best suited time units.
    """
    _sort_units = ('YEARS', 'DAYS', 'HOURS', 'MINS', 'SECS')

    def __init__(self):
        self._points = []

    def __iter__(self):
        return iter(self._points)

    def __len__(self):
        return len(self._points)

    def append_interval(self, delta, units='SECS'):
        """Appends a new time point after delta time from last point.

        Parameters
        ----------
        delta : float
            Time interval which must be added.
        units : str
            Time units in which delta value is measured. Default: 'SECS'.
        """
        if delta <= 0:
            raise ValueError('Duration cannot be less than zero')
        if self._points:
            last = self._points[-1]
        else:
            last = 0
        self._points.append(last + delta * TIME_UNITS[units])

    def insert_point(self, time, units='SECS'):
        """Inserts a new time point into time series.

        Parameters
        ----------
        time : float
            Time point to be inserted.
        units : str
            Units in which time value is measured. Default: 'SECS'.

        Returns
        -------
        index : int
            Index of position, at which new time point is inserted.
        """
        time = time * TIME_UNITS[units]
        index = np.searchsorted(self._points, time)
        self._points.insert(index, time)
        return index

    def durations(self):
        """Gets an iterator over time intervals.

        Returns
        -------
        duration : float
            Duration of time interval.
        """
        if self._points:
            yield self._points[0]
        for i in range(1, len(self._points)):
            dur = self._points[i] - self._points[i-1]
            yield dur

    @classmethod
    def adjust_time(cls, time):
        for unit in cls._sort_units:
            d = time / TIME_UNITS[unit]
            if d >= 1:
                return d, unit
        return time, 'SECS'


class IrradiationProfile:
    """Describes irradiation and relaxation.

    Parameters
    ----------
    norm_flux : float
        Flux value for normalization.
    """
    def __init__(self, norm_flux=None):
        self._norm = norm_flux
        self._flux = []
        self._times = TimeSeries()
        self._record = []
        self._zero_index = 0

    def irradiate(self, flux, duration, units='SECS', record=None, nominal=False):
        """Adds irradiation step.

        Parameters
        ----------
        flux : float
            Flux value in neutrons per sq. cm per sec.
        duration : float
            Duration of irradiation step.
        units : str
            Units of duration. 'SECS' (default), 'MINS', 'HOURS', 'YEARS'.
        record : str
            Results record type: 'SPEC', 'ATOMS'. Default: None - no record.
        nominal : bool
            Indicate that this flux is nominal and will be used in normalization.
        """
        if record is None:
            record = ''
        elif record != 'ATOMS' and record != 'SPEC':
            raise ValueError('Unknown record')
        if flux < 0:
            raise ValueError('Flux cannot be less than zero')
        self._flux.append(flux)
        self._times.append_interval(duration, units=units)
        self._record.append(record)
        self._zero_index = len(self._times) - 1
        if nominal:
            self._norm = flux

    def measure_times(self):
        """Gets a list of times, when output is made.

        Returns
        -------
        times : list[float]
            Output times in seconds.
        """
        return list(self._times)

    def relax(self, duration, units='SECS', record=None):
        """Adds relaxation step.

        Parameters
        ----------
        duration : float
            Duration of irradiation step.
        units : str
            Units of duration. 'SECS' (default), 'MINS', 'HOURS', 'YEARS'.
        record : str
            Results record type: 'SPEC', 'ATOMS'. Default: None - no record.
        """
        if record is None:
            record = ''
        elif record != 'ATOMS' and record != 'SPEC':
            raise ValueError('Unknown record')
        if duration <= 0:
            raise ValueError('Duration cannot be less than zero')
        self._flux.append(0)
        self._times.append_interval(duration, units=units)
        self._record.append(record)

    def insert_record(self, record, time, units='SECS'):
        """Inserts extra observation point for specified time.

        Parameters
        ----------
        record : str
            Record type.
        time : float
            Time left from the profile start.
        units : str
            Time units.
        """
        index = self._times.insert_point(time, units=units)
        self._flux.insert(index, self._flux[index])
        self._record.insert(index, record)
        if index <= self._zero_index:
            self._zero_index += 1

    def output(self, nominal_flux=None):
        """Creates FISPACT output for the profile.

        Parameters
        ----------
        nominal_flux : float
            Nominal flux at point of interest.

        Returns
        -------
        text : str
            Output.
        """
        if self._norm is not None and nominal_flux is not None:
            norm_factor = nominal_flux / self._norm
        else:
            norm_factor = 1
        lines = []
        last_flux = 0
        for i, (flux, dur, rec) in enumerate(zip(self._flux, self._times.durations(), self._record)):
            cur_flux = flux * norm_factor
            if cur_flux != last_flux:
                lines.append('FLUX {0:.5}'.format(cur_flux))
            time, unit = self._times.adjust_time(dur)
            lines.append('TIME {0:.5} {1} {2}'.format(time, unit, rec))
            last_flux = cur_flux
            if i == self._zero_index:
                lines.append('FLUX 0')
                lines.append('ZERO')
        # if last_flux > 0:
        #    lines.append('FLUX 0')
        return lines


def run_activation(title, material, volume, spectrum, folder, verbose=True,
                   settings=None):
    """Runs activation calculations for single case.

    Parameters
    ----------
    title : str
        Title for the inventory.
    material : Material
        Material to be irradiated.
    volume : float
        Volume of the material.
    spectrum : (ebins, flux)
        Flux data. ebins - energy bins; flux - group fluxes for every bin.
    folder : str
        Folder, where input files have to be located. Calculations will be run
        in this folder. If no such folder exists, it will be created.
    verbose : bool
        Verbose output of calculations status. Default: True
    settings : FispactSettings
        Fispact inventory settings. Defaul: None.
    """
    path, tasks = _create_case_files(
        folder, spectrum[0], spectrum[1], {0: (material, volume)}, settings
    )
    _run_case(tasks, cwd=path, verbose=verbose)


def fetch_result(folder, inventory='inventory'):
    """Fetches results of a single inventory calculation.

    Parameters
    ----------
    folder : str
        Folder, where input files have to be located. Calculations will be run
        in this folder. If no such folder exists, it will be created.
    inventory : str
        Name of inventory. Default: 'inventory'.

    Returns
    -------
    result : list
        The result of calculations.
    """
    result = {}
    path = _fetch_folder(folder)

    def create_result_object(data_class, keyword, raw_data):
        data = data_class()
        for d in raw_data:
            data.add_frame(d[keyword], d['duration'])
        return data

    file_path = path / '{0}.tab1'.format(inventory)
    if file_path.exists():
        raw_data = read_fispact_tab(str(file_path))
        kwd = 'atoms'
        result[kwd] = create_result_object(IsotopeData, kwd, raw_data)
    file_path = path / '{0}.tab2'.format(inventory)
    if file_path.exists():
        raw_data = read_fispact_tab(str(file_path))
        kwd = 'activity'
        result[kwd] = create_result_object(IsotopeData, kwd, raw_data)
    file_path = path / '{0}.tab3'.format(inventory)
    if file_path.exists():
        raw_data = read_fispact_tab(str(file_path))
        kwd = 'ingestion'
        result[kwd] = create_result_object(IsotopeData, kwd, raw_data)

        raw_data = read_fispact_tab(str(file_path))
        kwd = 'inhalation'
        result[kwd] = create_result_object(IsotopeData, kwd, raw_data)
    file_path = path / '{0}.tab4'.format(inventory)
    if file_path.exists():
        raw_data = read_fispact_tab(str(file_path))
        kwd = 'spectrum'
        result[kwd] = create_result_object(GammaSpectra, kwd, raw_data)

        # TODO: Consider possibility of usage of 22-bin gamma energy groups.
        # result['ebins'] = EBINS_24
        # TODO: Add class to store scalar results.
        # result['a-energy'] = [d['a-energy'] for d in data]
        # result['b-energy'] = [d['b-energy'] for d in data]
        # result['g-energy'] = [d['g-energy'] for d in data]
        # result['fissions'] = [d['fissions'] for d in data]
    return result


def _mesh_fact(shape):
    return lambda: SparseData(shape=shape)


def prepare_mesh_container(fmesh, volumes, irr_profile, relax_profile, folder,
                           read_only, **kwargs):
    """Prepares container for acitvation calculation results.

    Returns
    -------
    path : Path
        Path to store results of calculations.
    result : dict
        Dictionary of results. Physical value -> timeframe -> cell -> voxel->
        value.
    element_keywords, value_keywords : set
        A set of physical quantities, which will be calculated.
    indices : set
        A set of indices of mesh voxels, that are not empty.
    """
    path = Path(folder)
    if path.exists() and not path.is_dir():
        raise FileExistsError("Such file exists but it is not folder.")
    elif not path.exists():
        if read_only:
            raise FileNotFoundError("Data directory not found")
        path.mkdir()

    result = {
        'time': irr_profile.measure_times() + relax_profile.measure_times(),
        'fmesh': fmesh,
        'zero': len(irr_profile.measure_times()),
        'volumes': volumes
        # Other are optional
    }

    cells = volumes.keys()

    factory = _mesh_fact(fmesh.mesh.shape)

    element_keywords = set()
    value_keywords = set()
    if kwargs.get('tab1', False):
        result['atoms'] = [
            {c: defaultdict(factory) for c in cells} for t in result['time']
        ]
        element_keywords.add('atoms')
    if kwargs.get('tab2', False):
        result['activity'] = [
            {c: defaultdict(factory) for c in cells} for t in result['time']
        ]
        element_keywords.add('activity')
    if kwargs.get('tab3', False):
        result['ingestion'] = [
            {c: defaultdict(factory) for c in cells} for t in result['time']
        ]
        result['inhalation'] = [
            {c: defaultdict(factory) for c in cells} for t in result['time']
        ]
        element_keywords.add('ingestion')
        element_keywords.add('inhalation')
    if kwargs.get('tab4', False):
        result['ebins'] = EBINS_24
        result['spectrum'] = [
            {c: factory() for c in cells} for t in result['time']
        ]
        result['a-energy'] = [
            {c: factory() for c in cells} for t in result['time']
        ]
        result['b-energy'] = [
            {c: factory() for c in cells} for t in result['time']
        ]
        result['g-energy'] = [
            {c: factory() for c in cells} for t in result['time']
        ]
        result['fissions'] = [
            {c: factory() for c in cells} for t in result['time']
        ]
        value_keywords = {'spectrum', 'a-energy', 'b-energy', 'g-energy',
                          'fissions'}

    indices = reduce(
        set.union, [set(vols._data.keys()) for vols in volumes.values()]
    )

    return path, result, element_keywords, value_keywords, indices


def _encode_materials(cells):
    """Enumerate all materials found in cells.

    Parameters
    ----------
    cells : list
        List of cells.

    Returns
    -------
    mat_codes : dict
        A dictionary of material codes (Material->int).
    """
    code = 1
    mat_codes = {}
    for c in sorted(cells, key=lambda x: x['name']):
        mat = c.material()
        if mat not in mat_codes.keys():
            mat_codes[mat] = code
            code += 1
    return mat_codes


def _create_case_files(folder, ebins, flux, materials, settings=None,
                       arrayx=None):
    """Creates a folder with inventory files for a particular flux.

    Parameters
    ----------
    folder : str
        Folder name.
    ebins : list[float]
        Energy bin boundaries.
    flux : list[float]
        Neutron flux.
    materials : dict
        A dictionary of name->(material, volume). name will be used in the
        inventory file name.
    settings : FispactSettings
        FISPACT inventory calculations settings.
    arrayx : str
        Name of arrayx file. If None, condense file will be
        created and shielded for calculations. Default: None.

    Returns
    -------
    path : Path
        Path to the folder with tasks.
    tasks : list[tuple(str)]
        A list of tuples (task_name, files). task_name - name of task to be
        run by FISPACT. files - name of file with FISPACT settings.
    """
    path = _fetch_folder(folder)
    tasks = []
    # Create all input files
    if arrayx is None:
        arrayx = 'ARRAYX'
        tasks.append((create_condense(cwd=path), arrayx))
    else:
        arrayx = os.path.relpath(arrayx, start=path)
    files = create_files(cwd=path, arrayx=arrayx)
    tasks.append(create_convert(ebins, flux, cwd=path))
    tasks.append((
        create_collapse(use_binary=settings.use_binary, cwd=path), files
    ))
    for name, (mat, vol) in materials.items():
        title = 'case {0}'.format(name)
        tasks.append((
            create_inventory(
                title, mat, vol, sum(flux), cwd=path, settings=settings,
                inventory='inventory_{0}'.format(name)
            ),
            files
        ))
    return path, tasks


def _run_case(tasks, cwd=None, verbose=True):
    """Runs FISPACT calculations for the specific case.

    Parameters
    ----------
    tasks : list[tuple[str]]
        List of inventory names.
    cwd : Path
        Working path. Folder, where files are located.
    verbose : bool
        Turns on verbose output. Default: True.
    """
    for input_file, files in tasks:
        run_fispact(input_file, files=files, cwd=cwd, verbose=verbose)


def _run_mesh_cases(condense_task, flux_tasks, cwd=Path(), verbose=True,
                    threads=1):
    """Runs FISPACT calculations for the mesh.

    Parameters
    ----------
    condense_task : tuple
        A tuple, (condense_name, condense_files).
    flux_tasks : list
        A list of tasks to be run. Every item is a tuple: (path, tasks), where
        path is a Path instance - working directory for tasks. tasks - a list
        of tuples - tasks to be run.
    cwd : Path
        Working directory. Default: current folder.
    verbose : bool
        Output verbosity. Default: True.
    threads : int
        The number of threads to execute.
    """
    condense_name, condense_files = condense_task
    run_fispact(condense_name, condense_files, cwd=cwd, verbose=verbose)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        pool.map(lambda x: _run_case(x[1], cwd=x[0], verbose=verbose), flux_tasks)


def _create_full_mesh_cases(fmesh, volumes, folder, settings=None):
    """Runs activation calculations for every mesh voxel.

    Parameters
    ----------
    fmesh : FMesh
        FMesh data.
    volumes : dict
        A dictionary of cell volumes: cell->SparseData.
    folder : str
        Path to folder with calculation results.
    settings : FispactSettings
        Settings for FISPACT inventory.

    Returns
    -------
    path : Path
        Path to the folder with tasks.
    condense_task : tuple
        (condense_name, condense_files).
    flux_tasks : list
        A list of (flux_path, tasks).
    """
    path = _fetch_folder(folder)

    files = create_files(cwd=path)
    condense_task = (create_condense(cwd=path), files)

    flux_tasks = []
    indices = reduce(set.union, map(lambda x: set(x.keys()), volumes.values()))
    for i, j, k in indices:
        ebins, flux, err = fmesh.get_spectrum_by_index((i, j, k))
        ebins[0] = 1.e-11

        materials = _get_materials_filling_voxel(volumes, i, j, k)
        folder = 'voxel_{0}_{1}_{2}'.format(i, j, k)

        flux_tasks.append(
            _create_case_files(
                folder, ebins, flux, materials, settings, arrayx='ARRAYX'
            )
        )
    return path, condense_task, flux_tasks


def _create_simple_mesh_cases(fmesh, volumes, folder, settings=None):
    """Creates files for superimposed activation calculations.

    Parameters
    ----------
    fmesh : FMesh
        FMesh data.
    volumes : dict
        A dictionary of cell volumes: cell->SparseData.
    folder : str
        Path to folder with calculation results.
    settings : FispactSettings
        Settings for FISPACT inventory.

    Returns
    -------
    path : Path
        Path to the folder with tasks.
    condense_task : tuple
        (condense_name, condense_files).
    flux_tasks : list
        A list of (flux_path, tasks).
    """
    path = _fetch_folder(folder)

    files = create_files(cwd=path)
    condense_task = (create_condense(cwd=path), files)

    flux_tasks = []
    ebins = fmesh._ebins
    ebins[0] = 1.e-11
    max_flux = np.max(fmesh._data, axis=(1, 2, 3))
    max_volume = max([max(v._data.values()) for v in volumes.values() if v.size > 0])
    material_codes = _encode_materials(volumes.keys())
    materials = {name: (mat, max_volume) for mat, name in material_codes.items()}
    for i, f in enumerate(max_flux):
        flux = np.zeros_like(max_flux)
        flux[i] = f

        folder = 'ebin_{0}'.format(i)

        flux_tasks.append(
            _create_case_files(
                folder, ebins, flux, materials, settings, arrayx='ARRAYX'
            )
        )
    return path, condense_task, flux_tasks


def fetch_full_mesh_result(folder, volumes):
    """Fetches full mesh results.

    Parameters
    ----------
    folder : str
        Folder, that contains result.
    volumes : SparseData
        Cell volumes for every voxel.

    Returns
    -------
    timeframes : list
    """


def _get_materials_filling_voxel(volumes, i, j, k):
    materials = {}
    for c, vol in volumes.items():
        if vol[i, j, k] != 0 and c.material() is not None:
            materials[c['name']] = (c.material(), vol[i, j, k])
    return materials


def full_mesh_activation(title, fmesh, volumes, irr_profile, relax_profile,
                         folder, read_only=False, use_indices=None,
                         use_binary=False, **kwargs):
    """Do full calculations of activation for mesh voxels.

    Parameters
    ----------
    title : str
        Title for the inventory.
    fmesh : FMesh
        FMesh data.
    volumes : dict
        A dictionary of cell volumes: cell->SparseData.
    irr_profile : IrradiationProfile
        Irradiation profile
    relax_profile: IrradiationProfile
        Relaxation profile.
    folder : str
        Path to folder with calculation results.
    read_only : bool
        Only read already calculated results. Default: False.
    use_indices : Iterable
        List of voxel indices, where data must be calculated. Default: None -
        calculate for all non-empty voxels.
    use_binary : bool
        Use binary data rather text data for FISPACT data library.
    kwargs : dict
        Parameters for fispact inventory. See docs for fispact_inventory
        function.

    Returns
    -------
    result : dict
        A dictionary of calculation results. It contains the following keys:
        'time' - a list of time moments in seconds;
        'zero' - int - the starting index of relaxation phase.
        'volumes' - a dict of SparseData. cell->mesh of volumes [cc]
        'fmesh' - link to fmesh data.
        'ebins' - a list of gamma energy bins;
        All other data are lists - one item for each time moment.
        'atoms' - a of dict of dict of SparseData. material->isotope->mesh of concentrations [atoms];
        'activity' - a dict of dict of SparseData. material->isotope->mesh of activities [Bq];
        'ingestion' - a dict of dict of SparseData. material->isotope->mesh of ingestion dose [Sv/hour];
        'inhalation' - a dict of dict of SparseData. material->isotope->mesh of inhalation dose [Sv/hour]
        'spectrum' - a dict of SparseData. material->mesh of ndarrays [gammas/sec];
        'a-energy' - a dict of SparseData. material->mesh of alpha-activity [MeV/sec];
        'b-energy' - a dict of SparseData. material->mesh of beta-activity [MeV/sec];
        'g-energy' - a dict of SparseData. material->mesh of gamma-activity [MeV/sec];
        'fissions' - a dict of SparseData. material->mesh of spontaneous fission neutrons [neutrons/sec];
    """
    path, result, element_keywords, value_keywords, indices = \
        prepare_mesh_container(
            fmesh, volumes, irr_profile, relax_profile, folder, read_only,
            **kwargs
        )

    if use_indices is not None:
        indices = use_indices

    if not read_only:
        arrayx = str(path / 'arrayx')
        files = str(path / 'files')
        create_files(files=files, arrayx=arrayx)
        fispact_condense(condense=str(path / 'condense'), files=files)

    for i, j, k in indices:
        # cells, that are contained in this voxel.
        cells = {}
        for c, vol in volumes.items():
            if vol[i, j, k] != 0 and c.material() is not None:
                cells[c] = vol[i, j, k]

        if not cells:
            continue

        ebins, flux, err = fmesh.get_spectrum_by_index((i, j, k))
        ebins[0] = 1.e-11
        files = str(path / 'files_{0}_{1}_{2}'.format(i, j, k))
        if not read_only:
            fluxes = str(path / 'fluxes_{0}_{1}_{2}'.format(i, j, k))
            collapx = str(path / 'collapx_{0}_{1}_{2}'.format(i, j, k))

            create_files(files=files, collapx=collapx, fluxes=fluxes,
                         arrayx=arrayx)
            fispact_convert(ebins, flux, fluxes=fluxes)
            fispact_collapse(files=files, use_binary=use_binary)

        for c, vol in cells.items():
            mat = c.material()
            if not mat:
                continue
            inventory = str(
                path / 'inventory_{0}_{1}_{2}_b_{3}'.format(i, j, k, c['name'])
            )
            r = activation(
                title, mat, vol, (ebins, flux), irr_profile, relax_profile,
                inventory=inventory,
                files=files, overwrite=False, read_only=read_only, **kwargs
            )

            for key in value_keywords:
                for item, value in zip(result[key], r[key]):
                    item[c][i, j, k] = value
            for key in element_keywords:
                for item, r_item in zip(result[key], r[key]):
                    for elem, value in r_item.items():
                        item[c][elem][i, j, k] = value
    return result


def simple_mesh_activation(title, fmesh, volumes, irr_profile, relax_profile,
                           folder, read_only=False, check=None,
                           use_binary=False, **kwargs):
    """Do simplified calculations of activation for mesh voxels.

    Parameters
    ----------
    title : str
        Title for the inventory.
    fmesh : FMesh
        FMesh data.
    volumes : dict
        A dictionary of cell volumes: cell->SparseData.
    irr_profile : IrradiationProfile
        Irradiation profile
    relax_profile: IrradiationProfile
        Relaxation profile.
    folder : str
        Path to folder with calculation results.
    read_only : bool
        Only read already calculated results. Default: False.
    check : int
        The number of checks. Default: None - no checks.
    use_binary : bool
        Use binary data rather text data for FISPACT data library.
    kwargs : dict
        Parameters for fispact inventory. See docs for fispact_inventory
        function.

    Returns
    -------
    result : dict
        A dictionary of calculation results. It contains the following keys:
        'time' - a list of time moments in seconds;
        'zero' - int - the starting index of relaxation phase.
        'volumes' - a dict of SparseData. cell->mesh of volumes [cc]
        'fmesh' - link to fmesh data.
        'ebins' - a list of gamma energy bins;
        All other data are lists - one item for each time moment.
        'atoms' - a of dict of dict of SparseData. cell->isotope->mesh of concentrations [atoms];
        'activity' - a dict of dict of SparseData. cell->isotope->mesh of activities [Bq];
        'ingestion' - a dict of dict of SparseData. cell->isotope->mesh of ingestion dose [Sv/hour];
        'inhalation' - a dict of dict of SparseData. cell->isotope->mesh of inhalation dose [Sv/hour]
        'spectrum' - a dict of SparseData. cell->mesh of ndarrays [gammas/sec];
        'a-energy' - a dict of SparseData. cell->mesh of alpha-activity [MeV/sec];
        'b-energy' - a dict of SparseData. cell->mesh of beta-activity [MeV/sec];
        'g-energy' - a dict of SparseData. cell->mesh of gamma-activity [MeV/sec];
        'fissions' - a dict of SparseData. cell->mesh of spontaneous fission neutrons [neutrons/sec];
    """
    path, result, element_keywords, value_keywords, indices = \
        prepare_mesh_container(
            fmesh, volumes, irr_profile, relax_profile, folder, read_only,
            **kwargs
        )

    materials = set(c.material() for c in volumes.keys())
    materials.discard(None)

    files = str(path / 'files')
    if not read_only:
        arrayx = str(path / 'arrayx')
        create_files(files=files, arrayx=arrayx)
        fispact_condense(condense=str(path / 'condense'), files=files)

    ebins = fmesh._ebins
    ebins[0] = 1.e-11
    max_flux = np.max(fmesh._data, axis=(1, 2, 3))
    max_volume = max([max(v._data.values()) for v in volumes.values() if v.size > 0])
    for i, f in enumerate(max_flux):
        flux = np.zeros_like(max_flux)
        flux[i] = f
        if f != 0:
            factors = SparseData.from_dense(fmesh._data[i, :, :, :] / f)
        else:
            continue
        if not read_only:
            files = str(path / 'files_bin_{0}'.format(i))
            fluxes = str(path / 'fluxes_bin_{0}'.format(i))
            collapx = str(path / 'collapx_bin_{0}'.format(i))
            create_files(files=files, collapx=collapx, fluxes=fluxes,
                         arrayx=arrayx)
            fispact_convert(ebins, flux, fluxes=fluxes)
            fispact_collapse(files=files, use_binary=use_binary)

        mat_results = {}
        for m in materials:
            inventory = str(
                path / 'inventory_bin_{0}_c_{1}_d_{2:.4e}'.format(
                    i, m.composition._options['name'], m.density
                )
            )
            mat_results[m] = activation(
                title, m, max_volume, (ebins, flux), irr_profile, relax_profile, inventory=inventory,
                files=files, overwrite=False, read_only=read_only, **kwargs
            )

        # Result combination
        for c, volume in volumes.items():
            m = c.material()
            if m is None:
                continue
            for key in value_keywords:
                for item, value in zip(result[key], mat_results[m][key]):
                    item[c] += volume * factors * value / max_volume
            for key in element_keywords:
                for item, r_item in zip(result[key], mat_results[m][key]):
                    for elem, value in r_item.items():
                        item[c][elem] += volume * value * factors
    # Indices for result checking.
    # indices = random.sample(not_empty_indices, checks) if checks else []
    return result


def mesh_activation(title, fmesh, volumes, irr_profile, relax_profile, simple=True,
                    checks=None, folder=None, use_binary=False, read_only=False, **kwargs):
    """Runs activation calculations for mesh.

    There are two approaches. The first one is rigid - to calculate
    activation for all mesh voxels for all materials. But this approach is
    time consuming for large meshes. The second is to run one calculation
    for every material and for every energy group, and then to make a
    combination of the obtained results for every voxel. The latter approach
    may be inaccurate.

    Parameters
    ----------
    title : str
        Title for the inventory.
    fmesh : FMesh
        Neutron flux mesh data.
    volumes : dict
        Volumes of every cell in every mesh voxel. Body -> SparseData.
    irr_profile : IrradiationProfile
        Irradiation profile.
    relax_profile : IrradiationProfile
        Relaxation profile.
    simple : bool
        If True then simplified approach is used. Otherwise rigid approach is
        used. Default: True.
    checks : int
        The number of checks to be done for simplified approach. It is the number
        of rigid calculation to be done for random voxels just to compare the
        results. Default: None - no checks will be done.
    folder : str
        Name of output folder.
    use_binary : bool
        Use binary data rather text data.
    read_only : bool
        If the calculations have been already run and it is necessary only to read
        results, set this flag to True.
    kwargs : dict
        Parameters for fispact_inventory. See docs for fispact_inventory function.

    Returns
    -------
    result : dict
        A dictionary of calculation results. It contains the following keys:
        'time' - a list of time moments in seconds;
        'zero' - int - the starting index of relaxation phase.
        'volumes' - a dict of SparseData. cell->mesh of volumes [cc]
        'mesh' - mesh data.
        'ebins' - a list of gamma energy bins;
        All other data are lists - one item for each time moment.
        'atoms' - a of dict of dict of SparseData. cell->isotope->mesh of concentrations [atoms];
        'activity' - a dict of dict of SparseData. cell->isotope->mesh of activities [Bq];
        'ingestion' - a dict of dict of SparseData. cell->isotope->mesh of ingestion dose [Sv/hour];
        'inhalation' - a dict of dict of SparseData. cell->isotope->mesh of inhalation dose [Sv/hour]
        'spectrum' - a dict of SparseData. cell->mesh of ndarrays [gammas/sec];
        'a-energy' - a dict of SparseData. cell->mesh of alpha-activity [MeV/sec];
        'b-energy' - a dict of SparseData. cell->mesh of beta-activity [MeV/sec];
        'g-energy' - a dict of SparseData. cell->mesh of gamma-activity [MeV/sec];
        'fissions' - a dict of SparseData. cell->mesh of spontaneous fission neutrons [neutrons/sec];
    """
    if not simple:
        result = full_mesh_activation(title, fmesh, volumes, irr_profile,
                                      relax_profile, folder, read_only=read_only,
                                      use_binary=use_binary, **kwargs)
    else:
        result = simple_mesh_activation(title, fmesh, volumes, irr_profile,
                                        relax_profile, folder, read_only=read_only,
                                        check=checks, use_binary=use_binary, **kwargs)
    return result


def _fetch_folder(folder, read_only=False):
    path = Path(folder)
    if path.exists() and not path.is_dir():
        raise FileExistsError("Such file exists but it is not a folder.")
    elif not path.exists():
        if read_only:
            raise FileNotFoundError("Data directory not found")
        else:
            path.mkdir()
    return path

