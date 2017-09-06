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
    transform(tr)
        Applies transformation tr to this universe. Returns a new universe.
    """
    def __init__(self, cells, name=0, description=''):
        self.cells = tuple(cells)
        self.name = name
        self.description = description

    def transform(self, tr):
        tr_cells = []
        for cell in self.cells:
            tr_cells.append(cell.transform(tr))
        return Universe(tr_cells)
