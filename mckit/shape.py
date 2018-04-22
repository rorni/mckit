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

    def __new__(cls, opc, *args):
        cls._verify_opc(opc, *args)
        if opc == 'S' and isinstance(args[0], Shape):
            cls.already = True
            shape = args[0]

    def __init__(self, opc, *args):
        if not self.already:
            _Shape.__init__(opc, *args)
            self._hash = 0 # TODO: insert hash function

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
        if self.inverted_opc != other.opc:
            return False


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

