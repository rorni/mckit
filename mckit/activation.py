"""Module for activation calculations and FISPACT coupling."""
from itertools import accumulate
import subprocess
import numpy as np


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

TIME_UNITS = {'SECS': 1, 'MINS': 60, 'HOURS': 3600, 'DAYS': 3600*24, 'YEARS': 3600*24*365}


def fispact_files(files='files', collapx='COLLAPX', fluxes='fluxes', arrayx='ARRAYX'):
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
    """
    with open(files, mode='w') as f:
        for k, v in LIBS.items():
            f.write(k + '  ' + DATA_PATH + v + '\n')
        f.write('fluxes  ' + fluxes + '\n')
        f.write('collapxi  ' + collapx + '\n')
        f.write('collapxo  ' + collapx + '\n')
        f.write('arrayx  ' + arrayx + '\n')


def fispact_convert(ebins, flux, convert='convert.i', fluxes='fluxes', arb_flux='arb_flux', files='files.convert'):
    """Converts flux to the 709 groups.

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
    """
    with open(files, mode='w') as f:
        f.write('ind_nuc  ' + DATA_PATH + LIBS['ind_nuc'])
        f.write('fluxes  ' + fluxes + '\n')
        f.write('arb_flux  ' + arb_flux + '\n')

    with open(arb_flux, mode='w') as f:
        ncols = 6
        text = []
        for i, e in enumerate(ebins):
            s = '\n' if (i + 1) % ncols == 0 else ' '
            text.append('{0:.6e}'.format(e))
            text.append(s)
        text[-1] = '\n'
        f.write(''.join(text))

        text = []
        for i, e in enumerate(flux):
            s = '\n' if (i + 1) % ncols == 0 else ' '
            text.append('{0:.6e}'.format(e))
            text.append(s)
        text[-1] = '\n'
        f.write(''.join(text))
        f.write('{0}\n'.format(1))
        f.write('total flux={0:.6e}'.format(np.sum(flux)))

    with open(convert, mode='w') as f:
        text = [
            '<< convert flux to 709 grout structure >>'
            'CLOBBER',
            'GRPCONVERT {0} 709'.format(len(flux)),
            'FISPACT',
            '* SPECTRAL MODIFICATION'
            'END'
            '* END'
        ]
        f.write('\n'.join(text))

    status = subprocess.run('fispact', convert, files)
    status.check_returncode()
    return fluxes


def fispact_collapse(collapse='collapse.i', files='files', use_binary=True):
    """Collapses crossections with flux.

    Parameters
    ----------
    collapse : str
        Filename for collapse input file.
    files : str
        Filename for files input file.
    use_binary : bool
        Use binary data rather text data.
    """
    p = -1 if use_binary else +1
    with open(collapse, mode='w') as f:
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

    status = subprocess.run('fispact', collapse, files)
    status.check_returncode()


def fispact_condense(condense='condense.i', files='files'):
    """Condense the decay and fission data.

    Parameters
    ----------
    condense : str
        Name of condense input file.
    files : str
        Name of files input file.
    """
    with open(condense, mode='w') as f:
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

    status = subprocess.run('fispact', condense, files)
    status.check_returncode()


def fispact_inventory(title, material, volume, flux, irr_profile, relax_profile, inventory='inventory.i',
                      files='files', nat_reltol=1.e-8, zero=True, mind=1.e+5, use_fission=False, half=True,
                      hazards=False, tab1=False, tab2=False, tab3=False, tab4=False, nostable=False,
                      inv_tol=None, path_tol=None, uncertainty=0):
    """Runs inventory calculations.

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
    irr_profile : IrradiationProfile
        Irradiation profile.
    relax_profile : IrradiationProfile
        Relaxation profile.
    inventory : str
        File name for inventory input file.
    files : str
        File name for data file.
    nat_reltol : float
        Relative tolerance to believe that elements have natural abundance.
        To force use of isotopic composition set nat_reltol to None. Default: 1.e-8.
    zero : bool
        If True, then time value is reset to zero after an irradiation.
    mind : float
        Indicate the minimum number of atoms which are regarded as significant
        for the output inventory. Default: 1.e+5
    use_fission : bool
        Causes to use fission reactions. If it is False - fission reactions are omitted.
        Default: False - not to use fission.
    half : bool
        If True, causes the half-lije of each nuclide to be printed in the
        output at all timesteps. Default: True.
    hazards : bool
        If True, causes data on potential ingestion and inhalation doses to be read
        and dose due to individual nuclides to be printed at all timesteps.
        Default: False.
    tab1, tab2, tab3, tab4: bool
        If True, causes output of the specific data into separate files.
        tab1 - number of atoms and grams of each nuclide, default: False;
        tab2 - activity (Bq) and dose rate (Sv per hour) of each nuclide, default: False;
        tab3 - ingestion and inhalation dose (Sv) of each nuclide, default: False;
        tab4 - gamma-ray spectrum (MeV per sec) and the number of gammas per group, default: False.
    nostable : bool
        If True, printing of stable nuclides in the inventory is suppressed. Default: False
    inv_tol : (float, float)
        (atol, rtol) - absolute and relative tolerances for inventory calculations.
        Default: None - means default values ramain (1.e+4, 2.e-3).
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
    text.extend(fispact_material(material, volume, tolerance=nat_reltol))
    # Calculation parameters.
    text.append('MIND  {0:.5e}'.format(mind))
    if use_fission:
        text.append('USEFISSION')
    if half:
        text.append('HALF')
    if hazards:
        text.append('HAZARDS')
    if tab1:
        text.append('TAB1 1')
    if tab2:
        text.append('TAB2 1')
    if tab3:
        text.append('TAB3 1')
    if tab4:
        text.append('TAB4 1')
    if nostable:
        text.append('NOSTABLE')
    if inv_tol:
        text.append('TOLERANCE  0  {1:.4e}  {2:.4e}'.format(*inv_tol))
    if path_tol:
        text.append('TOLERANCE  1  {1:.4e}  {2:.4e}'.format(*path_tol))
    if uncertainty:
        text.append('UNCERTAINTY {0}'.format(uncertainty))
    # Irradiation and relaxation profiles
    text.extend(irr_profile.output(flux))
    if zero:
        text.append('ZERO')
    text.extend(relax_profile.output())
    # Footer
    text.append('END')
    text.append('* END of calculations')
    # Save to file
    with open(inventory, mode='w') as f:
        f.write('\n'.join(text))
    # Run calculations.
    status = subprocess.run('fispact', inventory, files)
    status.check_returncode()


def fispact_material(material, volume, tolerance=1.e-8):
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
    text = ['DENSITY {0}'.format(material.density())]
    composition = []
    if tolerance is not None:
        mat = material.natural(tolerance)
        if mat is not None:
            mass = volume * mat.density()
            for e in mat.elements():
                composition.append((e, mat.get_weight(e) * 100))
            text.append('MASS {0} {1}'.format(mass, len(composition)))
    else:
        mat = None

    if tolerance is None or mat is None:
        mat = material.expand()
        tot_atoms = volume * mat.concentration()
        for e in mat.elements():
            composition.append((e, mat.get_atomic(e) * tot_atoms))
        text.append('FUEL  {0}'.format(len(composition)))

    for e, f in sorted(composition, key=lambda x: -x[1]):
        text.append('  {0}   {1}'.format(e, f))
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


class IrradiationProfile:
    """Describes irradiation and relaxation.

    Parameters
    ----------
    norm_flux : float
        Flux value for normalization.
    """
    _sort_units = ('YEARS', 'DAYS', 'HOURS', 'MINS', 'SECS')

    def __init__(self, norm_flux=None):
        self._norm = norm_flux
        self._flux = []
        self._duration = []
        self._record = []

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
        self._flux.append(flux * self._norm)
        self._duration.append(duration * TIME_UNITS[units])
        self._record.append(record)
        if nominal:
            self._norm = flux

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
        self._flux.append(0)
        self._duration.append(duration * TIME_UNITS[units])
        self._record.append(record)

    def adjust_time(self, time):
        for unit in self._sort_units:
            d = time / TIME_UNITS[unit]
            if d > 1:
                return d, unit
        return time, ''

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
        time = time * TIME_UNITS[units]
        cum_time = accumulate(self._duration)
        index = np.searchsorted(cum_time, time)
        self._flux.insert(index, 0)
        self._record.insert(index, record)
        if index < len(cum_time):
            delta = cum_time[index] - time
            self._duration.insert(index, self._duration[index] - delta)
            self._duration[index + 1] = delta
        elif index > 0:
            delta = time - cum_time[index - 1]
            self._duration.append(delta)
        else:
            self._duration.append(time)

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
        if self._norm is not None:
            norm_factor = nominal_flux / self._norm
        else:
            norm_factor = 1
        lines = []
        last_flux = 0
        for flux, dur, rec in zip(self._flux, self._duration, self._record):
            cur_flux = flux * norm_factor
            if cur_flux != last_flux:
                lines.append('FLUX {0}'.format(cur_flux))
            time, unit = self.adjust_time(dur)
            lines.append('TIME {0} {1} {2}'.format(time, unit, rec))
            last_flux = cur_flux
        if last_flux > 0:
            lines.append('FLUX 0')
        return '\n'.join(lines)


def activation(title, material, volume, spectrum, irr_profile, relax_profile, inventory='inventory.i',
               files='files', fluxes='fluxes', collapx='COLLAPX', arrayx='ARRAYX', use_binary=False, **kwargs):
    """Runs activation calculations.

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
    irr_profile : IrradiationProfile
        Irradiation profile.
    relax_profile : IrradiationProfile
        Relaxation profile.
    inventory : str
        File name for inventory input file.
    files : str
        File name for data file.
    collapx : str
        Name of file of the collapsed cross sections. Default: COLLAPX
    fluxes : str
        Name of file with flux data. Default: fluxes.
    arrayx : str
        Name of arrayx file. Usually it is needed to be calculated only once.
    use_binary : bool
        Use binary data rather text data.
    kwargs : dict
        Paramters for fispact_inventory. See docs for fispact_inventory function.
    """
    fispact_files(files=files, collapx=collapx, fluxes=fluxes, arrayx=arrayx)
    fispact_convert(spectrum[0], spectrum[1], fluxes=fluxes)
    fispact_condense(files=files)
    fispact_collapse(files=files, use_binary=use_binary)
    fispact_inventory(title, material, volume, np.sum(spectrum[1]), irr_profile, relax_profile, inventory=inventory,
                      **kwargs)



