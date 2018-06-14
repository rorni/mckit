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
    cells : iterable
        A list of cells this universe consist of. Among these cells can be
        cells of other universes. The most outer universe will be created.
        Inner universes will be created either, but they will be accessed
        via the outer.
        
    Methods
    -------
    fill(cell)
        Fills every cell that has fill option by cells of filling universe.
    transform(tr)
        Applies transformation tr to this universe. Returns a new universe.
    inner_universes()
        Gets all inner universes.
    get_surfaces()
        Gets all surfaces contained.
    """
    def __init__(self, cells, name=0, comment=None):
        self._name = name
        self._comment = comment
        created_universes = {}
        self._cells = set()
        for c in cells:
            u = c.get('U', 0)
            if u != name:
                continue
            fill = c.get('FILL')
            if fill is not None:
                uname = fill['universe']
                if uname not in created_universes.keys():
                    created_universes[uname] = Universe(cells, name=uname)
                fill['universe'] = created_universes[uname]
            self._cells.add(c)

    @property
    def name(self):
        return self._name

    def fill(self, cell=None, universe=None):
        """Fills cells that have fill option by filling universe cells.

        This procedure is applied recursively. The resulting universe does not
        contain cells that have fill option. If cell name is specified, then
        only specified cell is filled. This method modifies current universe.

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
