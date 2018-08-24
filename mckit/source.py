from functools import reduce

from .activation import EBINS_24
from .fmesh import SparseData


def produce_gamma_sdef(cell_volumes, gamma_spectrum):
    """Gets gamma source SDEF.

    Parameters
    ----------
    cell_volumes : dict
        A dictionary of cell volumes. Body -> SparseData.
    gamma_spectrum : dict
        A dictionary of gamma-source spectrum for every material.
        Material->SparseData.

    Returns
    -------
    total_gamma : float
        Total gamma intensity.
    distributions : list
        List of SDEF distributions.
    """
    # Drop void cells
    cell_volumes = {
        c: vols for c, vols in cell_volumes.items() if c.material is not None
    }

    distributions = [
        {'name': 1, 'var': ('L', []), 'prob': ('D', [])},   # cells
        {'name': 2, 'dep': ('S', [])},                      # x
        {'name': 3, 'dep': ('S', [])},                      # y
        {'name': 4, 'dep': ('S', [])},                      # z
        {'name': 5, 'dep': ('S', [])}                       # e
    ]

    dist_name = 6
    mesh = list(cell_volumes.values())[0].mesh

    dist_name, xdist, xi2dist = _create_bin_dists(mesh._xbins, dist_name)
    distributions.extend(xdist)

    dist_name, ydist, yi2dist = _create_bin_dists(mesh._ybins, dist_name)
    distributions.extend(ydist)

    dist_name, zdist, zi2dist = _create_bin_dists(mesh._zbins, dist_name)
    distributions.extend(zdist)

    dist_name, edist, ei2dist = _create_bin_dists(EBINS_24, dist_name)
    distributions.extend(edist)

    gamma_intensity = {}
    for cell, vols in cell_volumes.items():
        mat = cell.material
        gamma_intensity[cell] = gamma_spectrum[mat] * vols

    total_gamma = reduce(sum, map(SparseData.total, gamma_intensity.values()))

    for cell, intensity in gamma_intensity.items():
        name = cell['name']
        probs = intensity / total_gamma
        for (i, j, k), eprob in probs:
            for e, p in enumerate(eprob):
                if p != 0.0:
                    distributions[0]['var'][1].append(name)
                    distributions[0]['prob'][1].append(p)
                    distributions[1]['dep'][1].append(xi2dist[i])
                    distributions[2]['dep'][1].append(yi2dist[j])
                    distributions[3]['dep'][1].append(zi2dist[k])
                    distributions[4]['dep'][1].append(ei2dist[e])

    return total_gamma, distributions


def _create_bin_dists(bins, name):
    """Creates bin distributions.

    Parameters
    ----------
    bins : array_like[float]
        Bin boundaries.
    name : int
        The name of first distribution.

    Returns
    -------
    free_name : int
        Next available name.
    distributions : list
        List of new distributions.
    bin2name : dict
        Mapping from bin index to distribution number.
    """
    distributions = []
    bin2name = {}
    for i in range(len(bins) - 1):
        distributions.append(
            {'name': name, 'var': ('H', [bins[i], bins[i+1]]),
             'prob': ('D', [0.0, 1.0])}
        )
        bin2name[i] = name
        name += 1
    return name, distributions, bin2name

