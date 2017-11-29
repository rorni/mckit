# -*- coding: utf-8 -*-

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
    get_volumes(cell_names)
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

    def get_volumes(self, cell_names=None):
        """Calculates volumes of cells.

        Parameters
        ----------
        cell_names : list[int]
            Names of cells which volumes are to be calculated.

        Returns
        -------
        volumes : dict or list
            Volumes of cells
        """
        raise NotImplementedError

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
