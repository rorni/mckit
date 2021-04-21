import typing as tp

from typing import Iterable, List, NewType, Optional, Set, Union

import os

from copy import deepcopy
from functools import reduce
from itertools import groupby, permutations, product
from multiprocessing import Pool

import mckit.material as mm
import numpy as np

from click import progressbar

from .box import GLOBAL_BOX, Box
from .card import Card
from .constants import MIN_BOX_VOLUME

# noinspection PyUnresolvedReferences,PyPackageRequirements
from .geometry import Shape as _Shape
from .printer import CELL_OPTION_GROUPS, print_card, print_option
from .surface import Surface
from .transformation import Transformation
from .utils import filter_dict

__all__ = ["Shape", "Body", "simplify", "GLOBAL_BOX", "Card", "TGeometry", "TGeometry"]


# noinspection PyProtectedMember
class Shape(_Shape):
    """Describes shape.

    Shape is immutable object.

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

    Properties
    ----------
    opc : str
        Operation code. It may be different from opc passed in __init__.
    invert_opc : str
        Operation code, complement to the opc.
    args : tuple
        A tuple of shape's arguments.

    Methods
    -------
    test_box(box)
        Tests if the box intersects the shape.
    volume(box, min_volume)
        Calculates the volume of the shape with desired accuracy.
    bounding_box(box, tol)
        Finds bounding box for the shape with desired accuracy.
    test_points(points)
        Tests the senses of the points.
    is_complement(other)
        Checks if other is a complement to the shape.
    complement()
        Gets complement shape.
    union(*other)
        Creates an union of the shape with the others.
    intersection(*other)
        Creates an intersection of the shape with the others.
    transform(tr)
        Gets transformed version of the shape.
    is_empty()
        Checks if this shape is empty - no space belong to it.
    get_surfaces()
        Gets all Surface objects that bounds the shape.
    complexity()
        Gets the complexity of the shape description.
    get_simplest()
        Gets the simplest description of the shape.
    replace_surfaces(replace_dict)
        Creates new Shape object by replacing surfaces.
    """

    _opc_hash = {
        "I": hash("I"),
        "U": ~hash("I"),
        "E": hash("E"),
        "R": ~hash("E"),
        "S": hash("S"),
        "C": ~hash("S"),
    }

    def __init__(self, opc, *args):
        opc, args = Shape._clean_args(opc, *args)
        _Shape.__init__(self, opc, *args)
        self._calculate_hash(opc, *args)

    def __iter__(self):
        return iter(self.args)

    def __getstate__(self):
        return self.opc, self.args, self._hash

    def __setstate__(self, state):
        opc, args, hash_value = state
        _Shape.__init__(self, opc, *args)
        self._hash = hash_value

    def __str__(self):
        words = print_card(self._get_words("U"))
        return words

    def __repr__(self):
        return f"Shape({self.opc}, {self.args})"

    def _get_words(self, parent_opc=None):
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
        if self.opc == "S":
            words.append("{0}".format(self.args[0].options["name"]))
        elif self.opc == "C":
            words.append("-{0}".format(self.args[0].options["name"]))
        elif self.opc == "E":
            words.append("EMPTY_SET")
        elif self.opc == "R":
            words.append("UNIVERSE_SET")
        else:
            sep = " " if self.opc == "I" else ":"
            args = self.args
            if self.opc == "U" and parent_opc == "I":
                words.append("(")
            for a in args[:-1]:
                words.extend(a._get_words(self.opc))
                words.append(sep)
            words.extend(args[-1]._get_words(self.opc))
            if self.opc == "U" and parent_opc == "I":
                words.append(")")
        return words

    @classmethod
    def _clean_args(cls, opc, *args):
        """Performs cleaning of input arguments."""
        args = [a.shape if isinstance(a, Body) else a for a in args]
        cls._verify_opc(opc, *args)
        if opc == "I" or opc == "U":
            args = [Shape("S", a) if isinstance(a, Surface) else a for a in args]
        if len(args) > 1:
            # Extend arguments
            args = list(args)
            i = 0
            while i < len(args):
                if args[i].opc == opc:
                    a = args.pop(i)
                    args.extend(a.args)
                else:
                    i += 1

            i = 0
            while i < len(args):
                a = args[i]
                if a.opc == "E" and opc == "I" or a.opc == "R" and opc == "U":
                    return a.opc, []
                elif a.opc == "E" and opc == "U" or a.opc == "R" and opc == "I":
                    args.pop(i)
                    continue
                for j, b in enumerate(args[i + 1 :]):
                    if a.is_complement(b):
                        if opc == "I":
                            return "E", []
                        else:
                            return "R", []
                i += 1
            args = list(set(args))
            args.sort(key=hash)
            if len(args) == 0:
                opc = "E" if opc == "U" else "R"
        if (
            len(args) == 1
            and isinstance(args[0], Shape)
            and (opc == "S" or opc == "I" or opc == "U")
        ):
            return args[0].opc, args[0].args
        elif len(args) == 1 and isinstance(args[0], Shape) and opc == "C":
            item = args[0].complement()
            return item.opc, item.args
        return opc, args

    @staticmethod
    def _verify_opc(opc, *args):
        """Checks if such argument combination is valid."""
        if (opc == "E" or opc == "R") and len(args) > 0:
            raise ValueError("No arguments are expected.")
        elif (opc == "S" or opc == "C") and len(args) != 1:
            raise ValueError("Only one operand is expected.")
        elif opc == "I" or opc == "U":
            if len(args) == 0:
                raise ValueError("Operands are expected.")

    def __eq__(self, other):
        if self is other:
            return True
        if self.opc != other.opc:
            return False
        if self.opc == "E" or self.opc == "R":
            return True
        if len(self.args) != len(other.args):
            return False
        self_groups = {
            k: list(v) for k, v in groupby(sorted(self.args, key=hash), key=hash)
        }
        other_groups = {
            k: list(v) for k, v in groupby(sorted(other.args, key=hash), key=hash)
        }
        flag = False  # TODO dvp: check is this statement doesn't break tests
        for hash_value, entities in self_groups.items():
            flag = False
            if hash_value not in other_groups.keys():
                return False
            if len(entities) != len(other_groups[hash_value]):
                return False
            for other_entities in permutations(other_groups[hash_value]):
                for se, oe in zip(entities, other_entities):
                    if not (se == oe):
                        break
                else:
                    flag = True
                    break
            if not flag:
                return False

        if flag:
            return True
        return False

    def __hash__(self):
        return self._hash

    def _calculate_hash(self, opc, *args):
        """Calculates hash value for the object.

        Hash is 'xor' for hash values of all arguments together with opc hash.
        """
        if opc == "C":  # C and S can be present only with Surface instance.
            self._hash = ~hash(args[0])
        elif opc == "S":
            self._hash = hash(args[0])
        else:
            self._hash = self._opc_hash[opc]
            for a in args:
                self._hash ^= hash(a)

    def complement(self):
        """Gets complement to the shape.

        Returns
        -------
        comp_shape : Shape
            Complement shape.
        """
        opc = self.opc
        args = self.args
        if opc == "S":
            return Shape("C", args[0])
        elif opc == "C":
            return Shape("S", args[0])
        elif opc == "E":
            return Shape("R")
        elif opc == "R":
            return Shape("E")
        else:
            opc = self.invert_opc
            c_args = [a.complement() for a in args]
            return Shape(opc, *c_args)

    def is_complement(self, other):
        """Checks if this shape is complement to the other.

        Returns
        -------
        result : bool
            Test result.
        """
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

    def intersection(self, *other):
        """Gets intersection with other shape.

        Parameters
        ----------
        other : tuple
            A list of Shape or Body objects, which must be intersected.

        Returns
        -------
        result : Shape
            New shape.
        """
        return Shape("I", self, *other)

    def union(self, *other):
        """Gets union with other shape.

        Parameters
        ----------
        other : tuple
            A list of Shape or Body objects, which must be joined.

        Returns
        -------
        result : Shape
            New shape."""
        return Shape("U", self, *other)

    def transform(self, tr):
        """Transforms the shape.

        Parameters
        ----------
        tr : Transformation
            Transformation to be applied.

        Returns
        -------
        result : Shape
            New shape.
        """
        opc = self.opc
        args = []
        for a in self.args:
            a = a.transform(tr)
            if isinstance(a, Surface):
                a = a.apply_transformation()
                # TODO dvp: check if call of apply_transformation() should be moved to caller site
                #           it would be better to change only transformations instead of the surfaces
            args.append(a)
        return Shape(opc, *args)

    def complexity(self):
        """Gets complexity of shape.

        Returns
        -------
        complexity : int
            The complexity of the shape description. It is the number of
            surfaces needed to describe the shape. Repeats are taken into
            account.
        """
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

    def get_surfaces(self) -> Set[Surface]:
        """Gets all the surfaces that describe the shape"""
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

    def is_empty(self):
        """Checks if the shape represents an empty set."""
        return self.opc == "E"

    def split_shape(self):
        shape_groups = []
        if self.opc == "U":
            stat = self.get_stat_table()
            drop_index = np.nonzero(np.all(stat == -1, axis=1))[0]
            arg_results = np.delete(stat, drop_index, axis=0)
            index_groups = self._find_groups(arg_results == +1)
            for ig in index_groups:
                index = np.nonzero(ig)[0]
                args = [self.args[i] for i in index]
                shape_groups.append(Shape("U", *args))
        elif self.opc == "I":
            arg_groups = [arg.split_shape() for arg in self.args]
            for args in product(*arg_groups):
                shape_groups.append(Shape("I", *args))
        else:
            shape_groups.append(self)
        return shape_groups

    @staticmethod
    def _find_groups(result):
        groups = [result[i, :] for i in range(result.shape[0])]
        while True:
            index = len(groups) - 1
            for j in range(index - 1, -1, -1):
                if np.any(groups[index] & groups[j] == 1):
                    groups[j] |= groups[index]
                    groups.pop(index)
                    break
            else:
                break
        return groups

    def get_simplest(self, trim_size=0):
        """Gets the simplest found description of the shape.

        Parameters
        ----------
        trim_size : int
            Shape variants with complexity greater than minimal one more than
            trim_size are thrown away.

        Returns
        -------
        shapes : list[Shape]
            A list of shapes with minimal complexity.
        """
        if self.opc != "I" and self.opc != "U":
            return [self]
        node_cases = []
        complexities = []
        stat = self.get_stat_table()
        if self.opc == "I":
            val = -1
        elif self.opc == "U":
            val = +1
        else:
            return {self}

        drop_index = np.nonzero(np.all(stat == -val, axis=1))[0]
        if len(drop_index) == 0:
            if self.opc == "I":
                return [Shape("E")]
            if self.opc == "U":
                return [Shape("R")]
        arg_results = np.delete(stat, drop_index, axis=0)
        if arg_results.shape[0] == 0:
            if self.opc == "I":
                return [Shape("R")]
            if self.opc == "U":
                return [Shape("E")]
        cases = self._find_coverages(arg_results, value=val)
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
    def _find_coverages(results, value=+1):
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
                    sub_cases = Shape._find_coverages(reminder, value=value)
                    for s in sub_cases:
                        s.append(j)
                cases.extend(sub_cases)
        for c in cases:
            c.sort()
        return cases

    def replace_surfaces(self, replace_dict):
        """Creates new Shape instance by replacing surfaces.

        If shape's surface is in replace dict, it is replaced by surface in
        dictionary values. Otherwise, the original surface is used. But new
        shape is created anyway.

        Parameters
        ----------
        replace_dict : dict
            A dictionary of surfaces to be replaced.

        Returns
        -------
        shape : Shape
            New Shape object obtained by replacing certain surfaces.
        """
        if self.opc == "C" or self.opc == "S":
            arg = self.args[0]
            surf = replace_dict.get(arg, arg)
            return Shape(self.opc, surf)
        elif self.opc == "I" or self.opc == "U":
            args = [arg.replace_surfaces(replace_dict) for arg in self.args]
            return Shape(self.opc, *args)
        else:
            return self

    @staticmethod
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
        for i, op in enumerate(polish):
            if isinstance(op, Surface):
                operands.append(Shape("S", op))
            elif isinstance(op, Shape):
                operands.append(op)
            elif op == "C":
                operands.append(operands.pop().complement())
            else:
                arg1 = operands.pop()
                arg2 = operands.pop()
                operands.append(Shape(op, arg1, arg2))
        return operands.pop()


TOperation = NewType("TOperation", str)
TGeometry = NewType("TGeometry", Union[List[Union[Surface, TOperation]], Shape, "Body"])


# noinspection PyProtectedMember
class Body(Card):
    """Represents MCNP cell.

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
    fill(universe)
        Fills this cell by universe.
    simplify(box, split_disjoint, min_volume)
        Simplifies cell description.
    transform(tr)
        Applies transformation tr to this cell.
    union(other)
        Returns an union of this cell with the other.
    """

    def __init__(self, geometry: TGeometry, **options: tp.Any) -> None:
        if isinstance(geometry, list):
            geometry = Shape.from_polish_notation(geometry)
        elif isinstance(geometry, Body):
            geometry = geometry.shape
        elif not isinstance(geometry, Shape):
            raise TypeError("Geometry list or Shape is expected.")
        Card.__init__(self, **options)
        self._shape = geometry

    def __repr__(self):
        options = str(self.options) if self.options else ""
        return f"Body({self._shape}, {options})"

    def __iter__(self):
        return iter(self._shape)

    def __hash__(self):
        return Card.__hash__(self) ^ hash(self._shape)

    def __eq__(self, other):
        return Card.__eq__(self, other) and self._shape == other._shape

    @property
    def transformation(self):
        return self.options.get("TRCL", None)

    def is_equivalent_to(self, other):
        result = self._shape == other._shape
        if result:
            if "FILL" in self.options:
                if "FILL" not in other.options:
                    return False
                my = self.options["FILL"]["universe"]
                their = other.options["FILL"]["universe"]
                return my.has_equivalent_cells(their)
        return result

    # TODO dvp: the method is used for printing, we'd better introduce virtual method print(self, out: TextIO)?
    # TODO dvp: in that case we could just return original text if available
    def mcnp_words(self, pretty=False):
        words = [str(self.name()), " "]
        if "MAT" in self.options.keys():
            words.append(str(self.options["MAT"].composition.name()))
            words.append(" ")
            words.append(str(-self.options["MAT"].density))
            words.append(" ")
        else:
            words.append("0")
            words.append(" ")
        words.extend(self._shape._get_words())
        words.append("\n")
        # insert options printing
        words.extend(self._options_list())
        for line in self.options.get("comment", []):
            words.append("$ " + str(line))
            words.append("\n")
        return words

    def _options_list(self):
        """Generates a list of option words. For __str__ method."""
        text = []
        for opt_group in CELL_OPTION_GROUPS:
            for key in opt_group:
                if key in self.options.keys():
                    text.extend(print_option(key, self.options[key]))
                    text.append(" ")
            text.append("\n")
        return text

    @property
    def shape(self):
        """Gets body's shape."""
        return self._shape

    def material(self) -> Optional[mm.Material]:
        """Gets body's Material. None is returned if no material present."""
        composition = self.options.get("MAT", None)
        assert composition is None or isinstance(composition, mm.Material)
        return composition

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
        geometry = self._shape.intersection(other)
        options = filter_dict(self.options, "original")
        return Body(geometry, **options)

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
        geometry = self._shape.union(other)
        options = filter_dict(self.options, "original")
        return Body(geometry, **options)

    def simplify(
        self,
        box=GLOBAL_BOX,
        split_disjoint=False,
        min_volume=MIN_BOX_VOLUME,
        trim_size=1,
    ):
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
        # print('Collect stage...')
        self._shape.collect_statistics(box, min_volume)
        # print('finding optimal solution...')
        variants = self._shape.get_simplest(trim_size)
        # print(len(variants))
        options = filter_dict(self.options, "original")
        return Body(variants[0], **options)

    def split(self, box=GLOBAL_BOX, min_volume=MIN_BOX_VOLUME):
        """Splits cell into disjoint cells.

        Returns
        -------
        cells : list

        """
        self.shape.collect_statistics(box, min_volume)
        shape_groups = self.shape.split_shape()
        bodies = [Body(shape, **self.options) for shape in shape_groups]
        return bodies

    # noinspection PyShadowingNames
    def fill(self, universe=None, recurrent=False, simplify=False, **kwargs):
        """Fills this cell by filling universe.

        If this cell doesn't contain fill options and universe does not
        specified, the cell itself is returned as list of length 1. Otherwise
        a list of cells from filling universe bounded by cell being filled is
        returned.

        Parameters
        ----------
        universe : Universe
            Universe which cells fill this one. If None, universe from 'FILL'
            option will be used. If no such universe, the cell itself will be
            returned. Default: None.
        recurrent : bool
            If filler universe also contains cells with fill option, they will
            be also filled. Default: False.
        simplify : bool
            If True, all cells obtained will be simplified.
        **kwargs : dict
            Keyword parameters for simplify method if simplify is True.
            Default: False.

        Returns
        -------
        cells : list[Body]
            Resulting cells.
        """
        if universe is None:
            if "FILL" in self.options.keys():
                universe = self.options["FILL"]["universe"]
                tr = self.options["FILL"].get("transform", None)
                if tr:
                    universe = universe.transform(tr)
            else:
                return [self]
        if recurrent:
            universe = universe.fill(recurrent=True, simplify=simplify, **kwargs)
        cells = []
        for c in universe:
            new_cell = c.intersection(self)  # because properties like MAT, etc
            # must be as in filling cell.
            if "U" in self.options.keys():
                new_cell.options["U"] = self.options["U"]  # except universe.
            if simplify:
                new_cell = new_cell.simplify(**kwargs)
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
        geometry = self._shape.transform(tr)
        options = filter_dict(self.options, "original")
        cell = Body(geometry, **options)
        fill = cell.options.get("FILL", None)
        if fill:
            tr_in = fill.get("transform", Transformation())
            new_tr = tr.apply2transform(tr_in)
            fill["transform"] = new_tr
        return cell

    def apply_transformation(self) -> "Body":
        """Actually apply transformation to this cell."""
        geometry = self._shape.apply_transformation()
        options = filter_dict(self.options, "original")
        cell = Body(geometry, **options)
        fill = cell.options.get("FILL", None)
        if fill:
            tr_in = fill.get("transform", None)
            filling_universe = fill["universe"]
            new_filling_universe = (
                deepcopy(filling_universe).transform(tr_in).apply_transformation()
            )
            cell.options["FILL"] = {"universe": new_filling_universe}
            # TODO dvp: this should create a lot of cell clashes on complex models with multiple filling with
            #           one universe. Should be resolved before saving.
        return cell


def simplify(
    cells: Iterable, box: Box = GLOBAL_BOX, min_volume: float = 1.0
) -> tp.Generator:
    """Simplifies the cells.

    Parameters
    ----------

    cells:
        iterable over cells to simplify
    box : Box
        Box, from which simplification process starts. Default: GLOBAL_BOX.
    min_volume : float
        Minimal volume of the box, when splitting process terminates.

    """

    for c in cells:
        cs = c.simplify(box=box, min_volume=min_volume)
        if not cs.shape.is_empty():
            yield cs


class Simplifier(object):
    def __init__(self, box: Box = GLOBAL_BOX, min_volume: float = 1.0):
        self.box = box
        self.min_volume = min_volume

    def __call__(self, cell: Body):
        return cell.simplify(box=self.box, min_volume=self.min_volume)

    def __getstate__(self):
        return self.box, self.min_volume

    def __setstate__(self, state):
        box, min_volume = state
        self.__init__(box, min_volume)


def simplify_mp(
    cells: Iterable[Body], box: Box = GLOBAL_BOX, min_volume: float = 1.0, chunksize=1
) -> tp.Generator:
    """Simplifies the cells in multiprocessing mode.

    Parameters
    ----------

    cells:
        iterable over cells to simplify
    box :
        Box, from which simplification process starts. Default: GLOBAL_BOX.
    min_volume : float
        Minimal volume of the box, when splitting process terminates.
    chunksize: size of chunks to pass to child processes
    """
    cpus = os.cpu_count()
    with Pool(processes=cpus) as pool:
        yield from pool.imap(
            Simplifier(box=box, min_volume=min_volume), cells, chunksize=chunksize
        )


def simplify_mpp(
    cells: Iterable[Body],
    box: Box = GLOBAL_BOX,
    min_volume: float = 1.0,
    chunksize: int = 1,
) -> tp.Generator:
    """Simplifies the cells in multiprocessing mode with progress bar.

    Parameters
    ----------

    cells:
        iterable over cells to simplify
    box :
        Box, from which simplification process starts. Default: GLOBAL_BOX.
    min_volume : float
        Minimal volume of the box, when splitting process terminates.
    chunksize: size of chunks to pass to child processes
    """

    def fmt_fun(x):
        return "Simplifying cell #{0}".format(x.name() if x else x)

    with progressbar(
        simplify_mp(cells, box, min_volume, chunksize), item_show_func=fmt_fun
    ) as pb:
        for c in pb:
            yield c
