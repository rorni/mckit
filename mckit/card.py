"""Features, common for all cards"""
from abc import ABC, abstractmethod

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


