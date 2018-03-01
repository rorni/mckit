# -*- coding: utf-8 -*-

import sys

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
    get_box_volumes(box, accuracy, pool_size, names)
        Calculates volumes of cells inside the box.
    get_mesh_volumes(mesh, accuracy, pool_size)
        Calculates volumes of cells inside mesh voxels.
    get_concentrations()
        Calculates concentrations of materials in voxels.
    populate()
        Populates every cell that has fill option by cells of filling universe.
    transform(tr)
        Applies transformation tr to this universe. Returns a new universe.
    """
    def __init__(self, cells, name=0, title=''):
        self.cells = tuple(cells)
        self.name = name
        self.description = title

    def get_mesh_volumes(self, mesh, accuracy=0.1, pool_size=20000):
        """Calculates volumes of cells that intersect the box or mesh voxels.

        Monte Carlo method of calculations is used.

        Parameters
        ----------
        mesh : RectMesh
            Mesh of calculations. Volumes of cells in every voxel are returned.
        accuracy : float
            Average linear density of random points (distance between two
            points). Given in cm. Default - 0.1 cm.
        pool_size : int
            The number of points generated at one iteration.
        
        Returns
        -------
        volumes : array[dict]
            Volumes of cells in every voxel. Array has the same shape as mesh.
            Every element of the array is a dictionary of cell_number -> 
            cell_volume_in_voxel pairs.
        """
        mesh_shape = mesh.shape()
        volumes = np.empty(shape=mesh_shape, dtype=dict)
        candidates = []
        for i, c in enumerate(self.cells):
            s = c.test_box(mesh)
            if s == +1 or s == 0:
                candidates.append(i)
        for i in range(mesh_shape[0]):
            for j in range(mesh_shape[1]):
                for k in range(mesh_shape[2]):
                    volumes[i, j, k] = self.get_box_volumes(
                        mesh.get_voxel(i, j, k),
                        accuracy=accuracy,
                        pool_size=pool_size,
                        names=candidates
                    )
        return volumes

    def get_box_volumes(self, box, accuracy=0.1, pool_size=100000, names=None,
                        verbose=False):
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
        names : list
            List of names of cells to be checked. If None, all cells are 
            checked.

        Returns
        -------
        volumes : dict
            Volumes of cells. The key is the index of cell.
        """
        volumes = {}
        checking_cells = names if names else list(range(len(self.cells)))
        tot_cells = len(checking_cells)
        if verbose:
            print("The number of cells: {0}".format(tot_cells))
        for i, name in enumerate(checking_cells):
            vol = self.cells[name].calculate_volume(box, accuracy=accuracy, pool_size=pool_size)
            if vol > 0:
                volumes[name] = vol
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write(
                    "cell={0} ({1}/{2}) ".format(name, i, tot_cells)
                )
                sys.stdout.flush()
        if verbose:
            print(volumes)
            print("\nDone")
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
        """Applies transformation tr to this universe. 
        
        Parameters
        ----------
        tr : Transformation
            Transformation to be applied.
            
        Returns
        -------
        universe : Universe
            New transformed universe.
        """
        tr_cells = []
        for cell in self.cells:
            tr_cells.append(cell.transform(tr))
        return Universe(tr_cells)

    def get_surfaces(self):
        """Gets all surfaces that discribe this universe.

        Returns
        -------
        surfaces : set[Surface]
            A set of all contained surfaces.
        """
        surfaces = set()
        for c in self.cells:
            cs = c.get_surfaces()
            surfaces.update(cs)
        return surfaces
