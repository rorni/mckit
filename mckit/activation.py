"""Module for activation calculations and FISPACT coupling."""
from itertools import accumulate
import re
import subprocess
import numpy as np

from .fispact_parser import read_fispact_tab
from .fmesh import ElementData, SparseData, SpectrumData

EBINS_24 = [
    0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.22,
    1.44, 1.66, 2.00, 2.50, 3.00, 4.00, 5.00, 6.50, 8.00, 10.0, 12.0, 14.0, 20.0
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

TIME_UNITS = {'SECS': 1., 'MINS': 60., 'HOURS': 3600., 'DAYS': 3600.*24, 'YEARS': 3600.*24*365}


class FispactError(Exception):
    pass


def fispact_fatal(text):
    """Raises FispactError exception if FATAL ERROR presents in output.

    Parameters
    ----------
    text : str
        Text to be checked.
    """
    match = re.search('^.*run +terminated.*$', text, flags=re.MULTILINE)
    if match:
        raise FispactError(match.group(0))


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


def fispact_convert(ebins, flux, convert='convert', fluxes='fluxes', arb_flux='arb_flux', files='files.convert'):
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
        f.write('ind_nuc  ' + DATA_PATH + LIBS['ind_nuc'] + '\n')
        f.write('fluxes  ' + fluxes + '\n')
        f.write('arb_flux  ' + arb_flux + '\n')

    with open(arb_flux, mode='w') as f:
        ncols = 6
        text = []
        for i, e in enumerate(reversed(ebins)):
            s = '\n' if (i + 1) % ncols == 0 else ' '
            text.append('{0:.6e}'.format(e * 1.e+6))  # Because fispact needs eV, not MeV
            text.append(s)
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

    with open(convert + '.i', mode='w') as f:
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

    status = subprocess.check_output(['fispact', convert, files], encoding='utf-8')
    print(status)
    fispact_fatal(status)


def fispact_collapse(collapse='collapse', files='files', use_binary=True):
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
    with open(collapse + '.i', mode='w') as f:
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

    status = subprocess.check_output(['fispact', collapse, files], encoding='utf-8')
    print(status)
    fispact_fatal(status)


def fispact_condense(condense='condense', files='files'):
    """Condense the decay and fission data.

    Parameters
    ----------
    condense : str
        Name of condense input file.
    files : str
        Name of files input file.
    """
    with open(condense + '.i', mode='w') as f:
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

    status = subprocess.check_output(['fispact', condense, files], encoding='utf-8')
    print(status)
    fispact_fatal(status)


def fispact_inventory(title, material, volume, flux, irr_profile, relax_profile, inventory='inventory',
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
        text.append('TOLERANCE  0  {1:.5e}  {2:.5e}'.format(*inv_tol))
    if path_tol:
        text.append('TOLERANCE  1  {1:.5e}  {2:.5e}'.format(*path_tol))
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
    with open(inventory + '.i', mode='w') as f:
        f.write('\n'.join(text))
    # Run calculations.
    status = subprocess.check_output(['fispact', inventory, files], encoding='utf-8')
    print(status)
    fispact_fatal(status)


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
    text = ['DENSITY {0}'.format(material.density)]
    composition = []
    if tolerance is not None:
        mat = material.composition.natural(tolerance)
        if mat is not None:
            mass = volume * mat.density / 1000    # Because mass must be specified in kg.
            for e in mat.composition.elements():
                composition.append((e, mat.composition.get_weight(e) * 100))
            text.append('MASS {0:.5} {1}'.format(mass, len(composition)))
    else:
        mat = None

    if tolerance is None or mat is None:
        mat = material.composition.expand()
        tot_atoms = volume * mat.concentration
        for e in mat.composition.elements():
            composition.append((e, mat.composition.get_atomic(e) * tot_atoms))
        text.append('FUEL  {0}'.format(len(composition)))

    for e, f in sorted(composition, key=lambda x: -x[1]):
        text.append('  {0}   {1:.5}'.format(e, f))
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
        self._flux.append(flux)
        self._duration.append(duration * TIME_UNITS[units])
        self._record.append(record)
        if nominal:
            self._norm = flux

    def measure_times(self):
        """Gets a list of times, when output is made.

        Returns
        -------
        times : list[float]
            Output times in seconds.
        """
        result = []
        time = 0
        for d, r in zip(self._duration, self._record):
            time += d
            if r is not None:
                result.append(time)
        return result

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
        if self._norm is not None and nominal_flux is not None:
            norm_factor = nominal_flux / self._norm
        else:
            norm_factor = 1
        lines = []
        last_flux = 0
        for flux, dur, rec in zip(self._flux, self._duration, self._record):
            cur_flux = flux * norm_factor
            if cur_flux != last_flux:
                lines.append('FLUX {0:.5}'.format(cur_flux))
            time, unit = self.adjust_time(dur)
            lines.append('TIME {0:.5} {1} {2}'.format(time, unit, rec))
            last_flux = cur_flux
        if last_flux > 0:
            lines.append('FLUX 0')
        return lines


def activation(title, material, volume, spectrum, irr_profile, relax_profile, inventory='inventory',
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
    fispact_inventory(title, material, volume, sum(spectrum[1]), irr_profile, relax_profile, inventory=inventory,
                      **kwargs)


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
    atom_frames : list[ElementData]
        List of ElementData instances for the number of atoms.
    activity_frames : list[ElementData]
        List of ElementData instances for the activities.
    ingestion_frames : list[Element_data]
        List of ElementData instances for the ingestion doses.
    spectrum_frames : list[SpectrumData]
        List of SpectrumData instances for the gamma spectra.
    alpha_data : list
        List of alpha activity data for each time frame. Each timeframe is
        a dictionary Body->SparseData.
    beta_data : list
        Beta activity data
    gamma_data : list
        Gamma activity_data
    fission_data : list
        Delayed fissions data.
    """
    times = irr_profile.measure_times() + relax_profile.measure_times()
    atom_frames = [ElementData(fmesh.mesh, t, units='at.') for t in times]
    activity_frames = [ElementData(fmesh.mesh, t, units='Bq') for t in times]
    ingestion_frames = [ElementData(fmesh.mesh, t, units='Sv/h') for t in times]
    spectrum_frames = [SpectrumData(fmesh.mesh, EBINS_24, t, volumes) for t in times]
    alpha_data = []
    beta_data = []
    gamma_data = []
    fission_data = []

    if simple:
        materials = set(c['MAT'] for c in volumes.keys() if c['MAT'] is not None)
        ebins, mean_flux = fmesh.mean_flux()

        if not read_only:
            # component calculation phase
            fispact_files()
            fispact_condense()

            for i, f in enumerate(mean_flux):
                flux = np.zeros_like(mean_flux)
                flux[i] = f

                files = 'files_bin_{0}'.format(i)
                fluxes = 'fluxes_bin_{0}'.format(i)
                collapx = 'collapx_bin_{0}'.format(i)
                fispact_files(files=files, collapx=collapx, fluxes=fluxes)
                fispact_convert(ebins, flux, fluxes=fluxes)
                fispact_collapse(files=files, use_binary=use_binary)

                for m in materials:
                    inventory = 'inventory_bin_{0}_mat_{1}'.format(i, m)
                    fispact_inventory(title, m, 1, sum(flux), irr_profile, relax_profile, inventory=inventory, **kwargs)

        # read results
        atom_data = []
        act_data = []
        ing_data = []
        gam_data = []
        for i, f in enumerate(mean_flux):
            atom_data.append({})
            act_data.append({})
            ing_data.append({})
            gam_data.append({})
            for m in materials:
                inventory = 'inventory_bin_{0}_mat_{1}'.format(i, m)
                if 'tab1' in kwargs.keys() and kwargs['tab1']:
                    atom_data[m] = read_fispact_tab(inventory + 'tab1')
                if 'tab2' in kwargs.keys() and kwargs['tab2']:
                    atom_data[m] = read_fispact_tab(inventory + 'tab2')
                if 'tab3' in kwargs.keys() and kwargs['tab3']:
                    atom_data[m] = read_fispact_tab(inventory + 'tab3')
                if 'tab4' in kwargs.keys() and kwargs['tab4']:
                    atom_data[m] = read_fispact_tab(inventory + 'tab4')

        # make superposition
        for cell, vol_mesh in volumes.items():
            mat = cell['MAT']
            if mat is None:
                continue
            for j in range(len(times)):
                for index, volume in vol_mesh:
                    ebins, flux, err = fmesh.get_spectrum_by_index(index)
                    factors = flux / mean_flux

                    for elem, amnt in atom_data[i][mat][j]['atoms'].items():
                        value = 0
                        for i, f in enumerate(factors):
                            value += amnt * f
                        atom_frames[j].add(cell, elem, index, value)

                    for elem, amnt in act_data[i][mat][j]['activity'].items():
                        value = 0
                        for i, f in enumerate(factors):
                            value += amnt * f
                        activity_frames[j].add(cell, elem, index, value)

                    for elem, amnt in ing_data[i][mat][j]['ingestion'].items():
                        value = 0
                        for i, f in enumerate(factors):
                            value += amnt * f
                        ingestion_frames[j].add(cell, elem, index, value)

                    value = np.zeros_like(EBINS_24)
                    for i, f in enumerate(factors):
                        value += np.array(gam_data[i][mat][j]['flux'])
                    spectrum_frames[j].add(cell, index, value)
