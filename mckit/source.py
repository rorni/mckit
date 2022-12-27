"""Code to load source distribution and print corresponding MCNP SDEF."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union

from functools import reduce

import numpy as np

from .printer import print_card, separate


class Distribution:
    """Represents a distribution of source variable.

    If the len of probs equals to the len of values, then the distribution is
    either of discrete variable (if is_discrete flag is set) or probs sets
    probability density values. If len(probs) + 1 == len(values), then
    values are bin boundaries for continuous distribution.

    Parameters
    ----------
    name : int
        Distribution's name. It is needed for MCNP.
    values : list
        A list of values, that variable can be equal to. Values can be a float,
        an int or other Distribution instances.
    probs : list or Distribution
        A list of probabilities or a distribution this one depends on.
        Probabilities need not be normalized.
    variable : str
        Source variable's name. Default: None.
    is_discrete : bool
        Indicate that the variable is discrete. Default: False.

    Methods
    -------
    mcnp_repr()
        Returns a string representation of corresponding MCNP card.
    get_inner()
        Gets all distributions this distribution depends on.
    depends_on()
        Gets distribution this one depends on.

    Properties
    ----------
    name : int
        Distribution's name.
    size : int
        Size of the distribution. It is the number of values or bins.
    variable : str
        Name of source variable.
    """

    def __init__(
        self,
        name: int,
        values: list[Union[int, float, "Distribution"]],
        probs: Union[List[float], "Distribution"],
        variable: Optional[str] = None,
        is_discrete: bool = False,
    ) -> None:
        self._name = name
        self._var = variable
        Distribution.check_distr(values, len(probs), is_discrete)
        self._is_discrete = is_discrete
        self._values = list(values)
        if isinstance(probs, Distribution):
            self._probs: Union[List[float], "Distribution"] = probs
        else:
            self._probs = list(probs)
        if (
            isinstance(self._probs, list)
            and len(self._probs) == len(values)
            and not is_discrete
            and not isinstance(self._values[0], Distribution)
        ):
            self._is_pdf = True
        else:
            self._is_pdf = False

    @property
    def name(self) -> int:
        """Gets distribution's name."""
        return self._name

    @property
    def size(self) -> int:
        """Gets distribution's size."""
        return len(self._probs)

    @property
    def variable(self) -> Optional[str]:
        """Gets distribution's source variable name."""
        return self._var

    def mcnp_repr(self) -> str:
        """Returns a string representation of corresponding MCNP card."""
        tokens = []
        discrete = self._is_discrete
        is_pdf = self._is_pdf
        if isinstance(self._values[0], Distribution):
            tokens.append("S")
            for v in self._values:
                tokens.append(str(v.name))
        else:
            if isinstance(self._values[0], list):
                values = reduce(list.__add__, self._values)
            else:
                values = self._values
            if discrete:
                tokens.append("L")
            elif is_pdf:
                tokens.append("A")
            else:
                tokens.append("H")
            for v in values:
                tokens.append(str(v))

        if isinstance(self._probs, Distribution):
            tokens.insert(0, f"DS{self._name}")
            prob_tokens = None
        else:
            tokens.insert(0, f"SI{self._name}")
            prob_tokens = [f"SP{self._name}"]
            if not is_pdf:
                prob_tokens.append("D")
            if len(self._values) == len(self._probs) + 1:
                prob_tokens.append("0")
            for p in self._probs:
                prob_tokens.append(str(p))
        card = print_card(separate(tokens))
        if prob_tokens:
            card += "\n" + print_card(separate(prob_tokens))
        return card

    def get_inner(self) -> Set["Distribution"]:
        """Gets nested distributions this one depends on.

        If values of this distribution are distributions themselves,
        then they are returned.

        Returns:
            Set:  A set of nested distributions.

        Raises:
            TypeError: when one of the values is not a `Distribution`.
        """
        dists: Set["Distribution"] = set()
        if isinstance(self._values[0], Distribution):
            for v in self._values:
                if not isinstance(v, Distribution):
                    raise TypeError(f"Expect `Distribution` type, actual {type(v)}")
                dists.add(v)
        return dists

    def __len__(self) -> int:
        """Size of distribution."""
        return len(self._probs)

    def depends_on(self) -> Optional["Distribution"]:
        """Gets distribution this one depends on.

        Returns:
            Distribution|None:  Distribution this one depends on, if any. None otherwise.
        """
        if isinstance(self._probs, Distribution):
            return self._probs
        else:
            return None

    @staticmethod
    def check_distr(
        values: list[Union[int, float, "Distribution"]], size: int, is_discrete: bool
    ) -> None:
        """Checks if the distribution is correct.

        Args:
            values: List of variable values.
            size: The length of intensity matrix along variable dimension.
            is_discrete:  True <=> the distribution is discrete.

        Raises:
            ValueError: if the distribution is neither discrete, nor pdf, nor histogram.
        """
        discr_or_pdf = len(values) == size
        histogram = len(values) == size + 1 and not is_discrete
        if not discr_or_pdf and not histogram:
            raise ValueError("Inconsistent size of values.")


def create_bin_distributions(
    bins: list[float], start_name: int = 1
) -> Tuple[int, list["Distribution"]]:
    """Creates bin distributions for specified bins.

    A list of distributions created. Index in the list corresponds to the  index of bin.

    Args:
        bins:   A list of bin boundaries.
        start_name:  Starting name for distributions.
                     For every new distribution the name is incremented by 1.

    Returns:
        free_name, list of distributions

    """
    distributions = []
    for low, high in zip(bins[:-1], bins[1:]):
        distributions.append(Distribution(start_name, [low, high], [1]))
        start_name += 1
    return start_name, distributions


def expand_matrix_distribution(
    intensities: np.ndarray,
    *var_values: list[float],
    start_name: int = 1,
) -> Tuple[int, Tuple[list[float], ...], List[float]]:
    """Converts matrix distribution to the len(var_values) linear.

    Creates:
        - expr_valu_values -  A tuple of lists of values for each source variable.
        - exp_intensities -   A list of expanded intensities for every hyperbin.

    Args:
        intensities :  A matrix of source intensities.
        var_values :   A tuple of variable values along each axis.
        start_name :   Starting name for distributions. Default: 1.

    Returns:
        free_name: int,  exp_var_values : tuple, exp_intensities : list[float]


    Raises:
        ValueError: if the distribution is neither discrete, nor pdf, nor histogram.
    """
    if len(intensities.shape) != len(var_values):
        msg = (
            "Length of var_values must be equal to the number of intensities dimensions."
        )
        raise ValueError(msg)
    uniq_values = []
    exp_var_values = []
    for dim, values in zip(intensities.shape, var_values):
        if Distribution.check_distr(values, dim):
            uniq_values.append(values)
        else:
            start_name, dists = create_bin_distributions(values, start_name)
            uniq_values.append(dists)
        exp_var_values.append([])

    exp_intensities = []
    for index, intensity in intensities:
        for j, i in enumerate(index):
            exp_var_values[j].append(uniq_values[j][i])
        exp_intensities.append(intensity)
    return start_name, tuple(exp_var_values), exp_intensities


class Source:
    """Represents particle source.

    Parameters
    ----------
    variables : dict
        A dictionary of source variables. var_name -> var_value. The value
        can be either float (int) value or Distribution instance.

    Methods
    -------
    mcnp_repr()
        Gets a string representation of corresponding MCNP card.
    """

    def __init__(self, **variables: Dict[int, Union[int, float, Distribution]]) -> None:
        self._variables = variables

    def mcnp_repr(self) -> str:
        """Gets a string representation of corresponding MCNP card."""
        tokens = ["SDEF"]
        cards = []
        extra_cards = []

        def _var_repr(value) -> str:
            if isinstance(value, Distribution):
                dep = value.depends_on()
                result = f"D{value.name}"
                if dep:
                    result = f"F{dep.variable} " + result
            else:
                result = str(value)
            return result

        for k, v in self._variables.items():
            tokens.append(f"{k}={_var_repr(v)}")
            if isinstance(v, Distribution):
                cards.append(v.mcnp_repr())
                for ec in sorted(v.get_inner(), key=lambda x: x.name):
                    extra_cards.append(ec.mcnp_repr())
        cards.insert(0, print_card(separate(tokens)))
        cards.extend(extra_cards)
        return "\n".join(cards)
