"""Features, common for all cards"""
from typing import Optional, Text, List
from abc import ABC, abstractmethod
from toolz import reduce
from operator import xor

from .printer import print_card


class Card(ABC):
    """Features, common for all cards."""
    def __init__(self, **options):
        self.options = options

    def name(self) -> Optional[int]:
        """Returns card's name."""
        return self.options.get('name', None)

    def rename(self, new_name) -> 'Card':
        """Renames the card.
        """
        self.options['name'] = new_name
        return self

    @property
    def is_anonymous(self):
        return not self.name()

    @abstractmethod
    def mcnp_words(self, pretty=False):
        """Gets a list of card words."""

    @property
    def has_original(self) -> bool:
        return 'original' in self.options.keys()

    @property
    def has_comment_above(self) -> bool:
        return "comment_above" in self.options.keys()

    def mcnp_repr(self, pretty: bool = False) -> List[Text]:
        """Gets str representation of the card."""
        # if self.has_original:
        #     if self.has_comment_above:
        #         return self.comment_above + '\n' + self.original
        #     else:
        #         return self.original
        # else:
        return print_card(self.mcnp_words(pretty))

    @property
    def comment_above(self) -> Optional[Text]:
        return self.options.get('comment_above', None)

    @property
    def original(self) -> Optional[Text]:
        return self.options.get('original', None)

    def drop_original(self) -> None:
        del self.options['original']

    def __str__(self):  # TODO dvp: option `name` is printed twice, should be explicit property of this class instance
        return "{}: \"{}\"".format(self.name(), self.options)

    def __hash__(self):  # TODO dvp: dict hashing implementation: check for effect of instability of the options
        return reduce(xor, map(lambda x:  hash(x[0]) ^ hash(x[1]), self.options.items()), 0)

    def __eq__(self, other):  # TODO dvp: what about nested dictionaries?
        for k in other.options.keys():
            if k not in self.options:
                return False
        for k, v in self.options.items():
            if k not in other.options:
                return False
            my = self.options[k]
            their = other.options[k]
            if my != their:
                return False
        return True




