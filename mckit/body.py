from functools import reduce
from itertools import product

import numpy as np

from .geometry import Shape as _Shape
from .surface import Surface
from .constants import GLOBAL_BOX, MIN_BOX_VOLUME
from .printer import print_card


class Shape(_Shape):
    """Describes shape.

    Parameters
    ----------
    opc : str
        Operation code. Denotes operation to be applied. Possible values:
        'I' - for intersection;
        'U' - for union;
        'C' - for complement;
        'S' - (same) no operation;
        'E' - empty set - no space occupied;
        'R' - whole space.
    args : list of Shape or Surface
        Geometry elements. It can be either Shape or Surface instances. But
        no arguments must be specified for 'E' or 'R' opc. Only one argument
        must present for 'C' or 'S' opc values.

    Returns
    -------
    shape : Shape
        Shape instance.
    """
    _opc_hash = {'I': hash('I'), 'U': ~hash('I'), 'E': hash('E'), 'R': ~hash('E'), 'S': hash('S'), 'C': ~hash('S')}

    def __init__(self, opc, *args):
        opc, args = Shape.clean_args(opc, *args)
        _Shape.__init__(self, opc, *args)
        self._calculate_hash(opc, *args)

    def __hash__(self):
        return self._hash

    def __str__(self):
        return print_card(self._get_words(None))

    def _get_words(self, parent_opc):
        """Gets list of words that describe the shape.

        Parameters
        ----------
        parent_opc : str
            Operation code of parent shape. It is needed for proper use of
            parenthesis.

        Returns
        -------
        words : list[str]
            List of words.
        """
        words = []
        if self.opc == 'S':
            words.append('{0}'.format(self.args[0].options['name']))
        elif self.opc == 'C':
            words.append('-{0}'.format(self.args[0].options['name']))
        elif self.opc == 'E':
            words.append('EMPTY_SET')
        elif self.opc == 'R':
            words.append('UNIVERSE_SET')
        else:
            sep = ' ' if self.opc == 'I' else ':'
            args = self.args
            if self.opc == 'U' and parent_opc == 'I':
                words.append('(')
            for a in args[:-1]:
                words.extend(a._get_words(self.opc))
                words.append(sep)
            words.extend(args[-1]._get_words(self.opc))
            if self.opc == 'U' and parent_opc == 'I':
                words.append(')')
        return words

    @classmethod
    def clean_args(cls, opc, *args):
        cls._verify_opc(opc, *args)
        if len(args) == 1 and isinstance(args[0], Shape) and (opc == 'S' or opc == 'I' or opc == 'U'):
            return args[0].opc, args[0].args
        elif len(args) == 1 and isinstance(args[0], Shape) and opc == 'C':
            item = args[0].complement()
            return item.opc, item.args
        elif len(args) > 1:
            # Extend arguments
            args = list(args)
            i = 0
            while i < len(args):
                if args[i].opc == opc:
                    a = args.pop(i)
                    args.extend(a.args)
                i += 1

            i = 0
            while i < len(args):
                a = args[i]
                if a.opc == 'E' and opc == 'I' or a.opc == 'R' and opc == 'U':
                    return 'E', []
                elif a.opc == 'E' and opc == 'U' or a.opc == 'R' and opc == 'I':
                    args.pop(i)
                    continue
                for j, b in enumerate(args[i + 1:]):
                    if a.is_complement(b):
                        if opc == 'I':
                            return 'E', []
                        else:
                            return 'R', []
                i += 1
            args.sort(key=hash)
        return opc, args

    def __eq__(self, other):
        if self is other:
            return True
        if self.opc != other.opc:
            return False
        if len(self.args) != len(other.args):
            return False
        for a, o in zip(self.args, other.args):
            if not (a == o):
                return False
        return True

    def complement(self):
        """Gets complement to the shape."""
        opc = self.opc
        args = self.args
        if opc == 'S':
            return Shape('C', args[0])
        elif opc == 'C':
            return Shape('S', args[0])
        elif opc == 'E':
            return Shape('R')
        elif opc == 'R':
            return Shape('E')
        else:
            opc = self.invert_opc
            c_args = [a.complement() for a in args]
            return Shape(opc, *c_args)

    def is_complement(self, other):
        """Checks if this shape is complement to the other."""
        if hash(self) != ~hash(other):
            return False
        if self.opc != other.invert_opc:
            return False
        if len(self.args) != len(other.args):
            return False
        if len(self.args) == 1:
            return self.args[0] == other.args[0]
        elif len(self.args) > 1:
            for a in self.args:
                for b in other.args:
                    if a.is_complement(b):
                        break
                else:
                    return False
        return True

    def _calculate_hash(self, opc, *args):
        """Calculates hash value for the object.

        Hash is 'xor' for hash values of all arguments together with opc hash.
        """
        if opc == 'C':  # C and S can be present only with Surface instance.
            self._hash = ~hash(args[0])
        elif opc == 'S':
            self._hash = hash(args[0])
        else:
            self._hash = self._opc_hash[opc]
            for a in args:
                self._hash ^= hash(a)

    def intersection(self, *other):
        """Gets intersection with other shape."""
        return Shape('I', self, *other)

    def union(self, *other):
        """Gets union with other shape."""
        return Shape('U', self, *other)

    def transform(self, tr):
        opc = self._opc
        args = []
        for a in self._args:
            args.append(a.transform(tr))
        return Shape(opc, *args)

    def complexity(self):
        """Gets complexity of shape."""
        args = self.args
        if len(args) == 1:
            return 1
        elif len(args) > 1:
            result = 0
            for a in args:
                result += a.complexity()
            return result
        else:
            return 0

    def get_surfaces(self):
        args = self.args
        if len(args) == 1:
            return {args[0]}
        elif len(args) > 1:
            result = set()
            for a in args:
                result = result.union(a.get_surfaces())
            return result
        else:
            return set()

    @staticmethod
    def _verify_opc(opc, *args):
        """Checks if such argument combination is valid."""
        if (opc == 'E' or opc == 'R') and len(args) > 0:
            raise ValueError("No arguments are expected.")
        elif (opc == 'S' or opc == 'C') and len(args) != 1:
            raise ValueError("Only one operand is expected.")
        elif opc == 'I' or opc == 'U':
            if len(args) == 0:
                raise ValueError("Operands are expected.")
            for a in args:
                if not isinstance(a, Shape):
                    print(type(a), opc)
                    raise TypeError("Shape instance is expected for 'I' and 'U' operations.")

    def get_simplest(self, trim_size=0):
        if self.opc != 'I' and self.opc != 'U':
            return [self]
        node_cases = []
        complexities = []
        stat = self.get_stat_table()
        print(stat)
        if self.opc == 'I':
            val = -1
        elif self.opc == 'U':
            val = +1
        else:
            return {self}

        drop_index = np.nonzero(np.all(stat == -val, axis=1))[0]
        if len(drop_index) == 0:
            if self.opc == 'I':
                return [Shape('E')]
            if self.opc == 'U':
                return [Shape('R')]
        arg_results = np.delete(stat, drop_index, axis=0)
        #print(arg_results)
        cases = self.find_coverages(arg_results, value=val)
        final_cases = set(tuple(c) for c in cases)
        if len(final_cases) == 0:
            print(self)
            return None
        unique = reduce(set.union, map(set, final_cases))
        args = self.args
        node_variants = {i: args[i].get_simplest(trim_size) for i in unique}
        for indices in final_cases:
            variants = [node_variants[i] for i in indices]
            for args in product(*variants):
                node = Shape(self.opc, *args)
                node_cases.append(node)
                complexities.append(node.complexity())
        sort_ind = np.argsort(complexities)
        final_nodes = []
        min_complexity = complexities[sort_ind[0]]
        for i in sort_ind:
            final_nodes.append(node_cases[i])
            if complexities[i] > min_complexity + trim_size:
                break
        return final_nodes

    @staticmethod
    def find_coverages(results, value=+1):
        n = results.shape[1]
        cnt = np.count_nonzero(results == value, axis=1)
        i = np.argmin(cnt)
        cases = []
        for j in range(n):
            if results[i][j] == value:
                reminder = np.compress(results[:, j] != value, results, axis=0)
                if reminder.shape[0] == 0:
                    sub_cases = [[j]]
                else:
                    sub_cases = Shape.find_coverages(reminder, value=value)
                    for s in sub_cases:
                        s = list(s)
                        s.append(j)
                cases.extend(sub_cases)
        for c in cases:
            c.sort()
        return cases


def from_polish_notation(polish):
    """Creates Shape instance from reversed Polish notation.

    Parameters
    ----------
    polish : list
        List of surfaces and operations written in reversed Polish Notation.

    Returns
    -------
    shape : Shape
        The geometry represented by Shape instance.
    """
    operands = []
    for op in polish:
        if isinstance(op, Surface):
            operands.append(Shape('S', op))
        elif op == 'C':
            operands.append(operands.pop().complement())
        else:
            arg1 = operands.pop()
            arg2 = operands.pop()
            operands.append(Shape(op, arg1, arg2))
    return operands.pop()


class Body(Shape):
    """Represents MCNP's cell.

    Parameters
    ----------
    geometry : list or Shape
        Geometry expression. It is either a list of Surface instances and
        operations (reverse Polish notation is used) or Shape object.
    options : dict
        A set of cell's options.

    Methods
    -------
    intersection(other)
        Returns an intersection of this cell with the other.
    populate(universe)
        Fills this cell by universe.
    simplify(box, split_disjoint, min_volume)
        Simplifies cell description.
    transform(tr)
        Applies transformation tr to this cell.
    union(other)
        Returns an union of this cell with the other.
    """
    def __init__(self, geometry, **options):
        if isinstance(geometry, list):
            geometry = from_polish_notation(geometry)
        Shape.__init__(self, geometry.opc, *geometry.args)
        self._options = dict(options)

    def __getitem__(self, key):
        return self._options[key]

    def __str__(self):
        text = [str(self['name']), ' ']
        if 'MAT' in self._options.keys():
            text.append(str(self['MAT']))
            text.append(' ')
            text.append(str(self['RHO']))
            text.append(' ')
        else:
            text.append('0')
            text.append(' ')
        text.extend(Shape._get_words(self, None))
        text.append('\n')
        # insert options printing
        return print_card(text)

    def intersection(self, other):
        """Gets an intersection if this cell with the other.

        Other cell is a geometry that bounds this one. The resulting cell
        inherits all options of this one (the caller).

        Parameters
        ----------
        other : Cell
            Other cell.

        Returns
        -------
        cell : Cell
            The result.
        """
        geometry = Shape.intersection(self, other)
        return Body(geometry, **self._options)

    def union(self, other):
        """Gets an union if this cell with the other.

        The resulting cell inherits all options of this one (the caller).

        Parameters
        ----------
        other : Cell
            Other cell.

        Returns
        -------
        cell : Cell
            The result.
        """
        geometry = Shape.union(self, other)
        return Body(geometry, **self._options)

    def simplify(self, box=GLOBAL_BOX, split_disjoint=False,
                 min_volume=MIN_BOX_VOLUME, trim_size=1):
        """Simplifies this cell by removing unnecessary surfaces.

        The simplification procedure goes in the following way.
        # TODO: insert brief description!

        Parameters
        ----------
        box : Box
            Box where geometry should be simplified.
        split_disjoint : bool
            Whether to split disjoint geometries into separate geometries.
        min_volume : float
            The smallest value of box's volume when the process of box splitting
            must be stopped.
        trim_size : int
            Max size of set to return. It is used to prevent unlimited growth
            of the variant set.

        Returns
        -------
        simple_cell : Cell
            Simplified version of this cell.
        """
        print('Collect stage...')
        self.collect_statistics(box, min_volume)
        print('finding optimal solution...')
        variants = self.get_simplest(trim_size)
        print(len(variants))
        return Body(variants[0], **self._options)

    def populate(self, universe=None):
        """Fills this cell by filling universe.

        If this cell doesn't contain fill options, the cell itself is returned
        as list of length 1. Otherwise a list of cells from filling universe
        bounded by cell being filled is returned.

        Parameters
        ----------
        universe : Universe
            Universe which cells fill this one. If None, universe from 'FILL'
            option will be used. If no such universe, the cell itself will be
            returned.

        Returns
        -------
        cells : list[Cell]
            Resulting cells.
        """
        if universe is None:
            if 'FILL' in self.keys():
                universe = self['FILL']
            else:
                return [self]
        cells = []
        for c in universe.cells:
            new_cell = c.intersection(self)  # because properties like MAT, etc
                                             # must be as in filling cell.
            if 'U' in self._options.keys():
                new_cell['U'] = self['U']    # except universe.
            cells.append(new_cell)
        return cells

    def transform(self, tr):
        """Applies transformation to this cell.

        Parameters
        ----------
        tr : Transform
            Transformation to be applied.

        Returns
        -------
        cell : Cell
            The result of this cell transformation.
        """
        geometry = self.transform(tr)
        return Body(geometry, **self._options)
