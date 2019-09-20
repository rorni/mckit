"""Features, common for all cards"""
from abc import ABC, abstractmethod
from toolz import reduce
from operator import xor

from .printer import print_card


class Card(ABC):
    """Features, common for all cards."""
    def __init__(self, **options):
        self.options = options

    def name(self):
        """Returns card's name."""
        return self.options.get('name', None)

    def rename(self, new_name):
        """Renames the card.

        Parameters
        ----------
        new_name : int
            New card's name. 
        """
        self.options['name'] = new_name

    @abstractmethod
    def mcnp_words(self):
        """Gets a list of card words."""

    def mcnp_repr(self):
        """Gets str representation of the card."""
        words = self.mcnp_words()
        return print_card(words)

    def __str__(self):
        return "{}: \"{}\"".format(self.name, self.options)

    def __hash__(self):
        return reduce(xor, map(lambda x:  hash(x[0]) ^ hash(x[1]), self.options.items()), 0)

    def __eq__(self, other):
        for k in other.options.keys():
            if k not in self.options:
                return False
        for k, v in self.options:
            if k not in other.options:
                return False
            my = self.options[k]
            their = other.options[k]
            if my != their:
                return False
        return True




