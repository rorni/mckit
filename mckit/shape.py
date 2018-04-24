from .geometry import Shape as _Shape
from .surface import Surface

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
    args : Shape or Surface
        Geometry elements. It can be either Shape or Surface instances. But
        no arguments must be specified for 'E' or 'R' opc. Only one argument
        must present for 'C' or 'S' opc values.

    Returns
    -------
    shape : Shape
        Shape instance.
    """
    already = False
    opc = None
    args = None
    inv_opc = {'I': 'U', 'E': 'R', 'C': 'S', 'U': 'I', 'R': 'E', 'S': 'C'}
    opc_hash = {'I': hash('I'), 'U': ~hash('I'), 'C': hash('C'), 'S': ~hash('C'), 'E': hash('E'), 'S': ~hash('E')}

    def __new__(cls, opc, *args):
        """Do argument reduction.

        It find what the best argument combination for the opc-args case supplied.
        It performs the following checking and operations:

        1. len(args) == 1 && args[0] - Shape --> args[0] is returned.
        2. One of the args is complement to some other. -->
           2.1. 'I' --> empty set.
           2.2. 'U' --> universe set.
        3. 'I'
           3.1. One of the arguments is Empty set --> Empty set.
           3.2. One of the arguments is Universe set ---> remove Universe set.
        4. 'U'
           4.1. One of the arguments is Empty set --> remove Empty set.
           4.2. One of the arguments is Universe set --> Universe set.

        Returns
        -------
        new_opc : str
            New opc.
        new_args : list
            new arguments.
        """
        cls._verify_opc(opc, *args)
        if len(args) == 1 and isinstance(args[0], Shape):
            cls.already = True
            return args[0]
        elif len(args) > 1:
            cls.args = []
            for i, a in enumerate(args[:-1]):
                if a.opc == 'E' and opc == 'I' or a.opc == 'R' and opc == 'U':
                    cls.already = True
                    return a
                elif a.opc == 'E' and opc == 'U' or a.opc == 'R' and opc == 'I':
                    continue
                for j, b in enumerate(args[i+1:]):
                    if a.is_complement(b):
                        cls.already = True
                        if opc == 'I':
                            return Shape('E')
                        else:
                            return Shape('R')
                cls.args.append(a)
            a = args[-1]
            if a.opc == 'E' and opc == 'I' or a.opc == 'R' and opc == 'U':
                cls.already = True
                return a
            elif not (a.opc == 'E' and opc == 'U' or a.opc == 'R' and opc == 'I'):
                cls.args.append(args[-1])
        cls.already = False
        return super().__new__()

    def __init__(self, opc, *args):
        if not self.already:
            if self.args:
                args = self.args
            _Shape.__init__(opc, *args)
            self._hash = self._calculate_hash() # TODO: insert hash function

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if self is other:
            return True
        if self.opc != other.opc:
            return False
        return self.args == other.args

    def complement(self):
        """Gets complement to the shape."""
        if self.opc == 'S':
            return Shape('C', self.args[0])
        elif self.opc == 'C':
            return Shape('S', self.args[0])
        elif self.opc == 'E':
            return Shape('R')
        elif self.opc == 'R':
            return Shape('E')
        else:
            if self.opc == 'I':
                opc = 'U'
            else:
                opc = 'I'
            args = [a.complement() for a in self.args]
            return Shape(opc, *args)

    def is_complement(self, other):
        """Checks if this shape is complement to the other."""
        if hash(self) != ~hash(other):
            return False
        if self.inverted_opc != other.opc:
            return False
        if len(self.args) != len(other.args):
            return False
        if len(self.args) == 1:
            return self.args[0] == other.args[0]
        elif len(self.args) > 1:
            for a, o in zip(self, reversed(other)):
                if not a.is_complement(o):
                    return False
        return True

    def _calculate_hash(self):
        """Calculates hash value for the object."""
        pass


    def intersection(self, *other):
        """Gets intersection with other shape."""
        return Shape('I', self, other)

    def union(self, other):
        """Gets union with other shape."""
        return Shape('U', self, other)

    @classmethod
    def _verify_opc(cls, opc, *args):
        """Checks if such argument combination is valid."""
        if (opc == 'E' or opc == 'R') and len(args) > 0:
            raise ValueError("No arguments are expected.")
        elif (opc == 'S' or opc == 'C') and len(args) != 1:
            raise ValueError("Only one operand is expected.")
        elif (opc == 'I' or opc == 'U') and len(args) == 0:
            raise ValueError("Operands are expected.")

