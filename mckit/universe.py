# -*- coding: utf-8 -*-

import numpy as np


class Universe:
    """Describes universe - a set of cells.
    
    Universe is a set of cells from which it consist of. Each cell can be filled
    with other universe. In this case, cells of other universe are bounded by
    cell being filled.
    
    Parameters
    ----------
    cells : list
        A list of cells this universe consist of.
        
    Methods
    -------
    get_volumes(region, cell_names)
        Calculates volumes of cells.
    get_concentrations()
        Calculates concentrations of materials in voxels.
    populate()
        Populates every cell that has fill option by cells of filling universe.
    transform(tr)
        Applies transformation tr to this universe. Returns a new universe.
    """
    def __init__(self, cells, name=0, description=''):
        self.cells = tuple(cells)
        self.name = name
        self.description = description

    def get_volumes(self, region, accuracy=0.1, pool_size=20000):
        """Calculates volumes of cells that intersect the box or mesh voxels.

        Monte Carlo method of calculations is used.

        Parameters
        ----------
        region : Box or RectMesh
            Region of calculations. If it is RectMesh instance, then volumes of
            cells in every voxel are returned.
        accuracy : float
            Average linear density of random points (distance between two
            points). Given in cm. Default - 0.1 cm.
        pool_size : int
            The number of points generated at one iteration.

        Returns
        -------
        volumes : dict
            Volumes of cells. The key is the index of cell.
        """

    def _get_box_volumes(self, box, accuracy=0.1, pool_size=20000):
        """Calculates volumes of cells that intersect the box.

        Monte Carlo method of calculations is used.

        Parameters
        ----------
        box : Box
            The box.
        accuracy : float
            Average linear density of random points (distance between two
            points). Given in cm. Default - 0.1 cm.
        pool_size : int
            The number of points generated at one iteration.

        Returns
        -------
        volumes : dict
            Volumes of cells. The key is the index of cell.
        """
        volumes = {}
        base_volume = box.volume()
        # First find cells that might intersect box.
        candidates = []
        for i, c in enumerate(self.cells):
            s = c.test_box(box)
            if s == +1:  # Box lies entirely inside cell c.
                return {i: base_volume}
            elif s == 0:
                candidates.append(i)
                volumes[i] = 0
        # Calculate volumes.
        n_points = max(int(np.ceil(base_volume / accuracy**3)), pool_size)
        n_repeat = int(np.ceil(n_points // pool_size))
        for i in range(n_repeat):
            points = box.random_points(pool_size)
            for name in candidates:
                cell_result = self.cells[name].test_point(points)
                volumes[name] += np.count_nonzero(cell_result == +1)
        for k in volumes.keys():
            volumes[k] *= base_volume / n_points
        return volumes

    def get_concentrations(self, mesh):
        """Calculates concentrations of each material in mesh.

        Parameters
        ----------
        mesh : Mesh
            Mesh object.

        Returns
        -------
        concentrations : dict
            Concentrations of materials for mesh.
        """
        raise NotImplementedError

    def populate(self):
        """Replaces cells that have fill option by filling universe cells.

        This procedure is applied recursively. The resulting universe does not
        contain cells that have fill option.

        Returns
        -------
        new_universe : Universe
            New universe that has no inner universes.
        """
        new_cells = []
        for c in self.cells:
            if 'FILL' in c.options.keys():
                new_cells += c.populate()
            else:
                new_cells.append(c)
        return Universe(new_cells)

    def transform(self, tr):
        tr_cells = []
        for cell in self.cells:
            tr_cells.append(cell.transform(tr))
        return Universe(tr_cells)
