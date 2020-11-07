# -*- coding: utf-8 -*-

from .printer import print_card, separate


class Distribution:
    """Represents a distribution of source variable.

    If the len of probs equals to the len of values, then the distribution of
    discrete variable is created. If len(probs) + 1 == len(values), then
    values are bin boundaries for continuous distribution.

    Parameters
    ----------
    name : int
        Distribution's name. It is needed for MCNP.
    values : list
        A list of values, that variable can be equal to. Values can be float,
        int and other Distribution instances.
    probs : list or Distribution
        A list of probabilities or a distribution this one depends on.
        Probabilities need not be normalized.
    variable : str
        Source variable's name. Default: None.

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

    def __init__(self, name, values, probs, variable=None):
        self._name = name
        self._var = variable
        Distribution.is_discrete(values, len(probs))
        self._values = list(values)
        self._probs = probs if isinstance(probs, Distribution) else list(probs)

    @property
    def name(self):
        """Gets distribution's name."""
        return self._name

    @property
    def size(self):
        """Gets distribution's size."""
        return len(self._probs)

    @property
    def variable(self):
        """Gets distribution's source variable name."""
        return self._var

    def mcnp_repr(self):
        """Returns a string representation of corresponding MCNP card."""
        tokens = []
        discrete = Distribution.is_discrete(self._values, len(self._probs))
        if isinstance(self._values[0], Distribution):
            tokens.append("S")
            for v in self._values:
                tokens.append(str(v.name))
        else:
            tokens.append("L" if discrete else "H")
            for v in self._values:
                tokens.append(str(v))

        if isinstance(self._probs, Distribution):
            tokens.insert(0, "DS{0}".format(self._name))
            prob_tokens = None
        else:
            tokens.insert(0, "SI{0}".format(self._name))
            prob_tokens = ["SP{0}".format(self._name), "D"]
            if not discrete:
                prob_tokens.append("0")
            for p in self._probs:
                prob_tokens.append(str(p))
        card = print_card(separate(tokens))
        if prob_tokens:
            card += "\n" + print_card(separate(prob_tokens))
        return card

    def get_inner(self):
        """Gets nested distributions this one depends on.

        If values of this distribution are distributions itself, then they
        are returned.

        Returns
        -------
        dists : set
            A set of nested distributions.
        """
        dists = set()
        if isinstance(self._values[0], Distribution):
            for v in self._values:
                dists.add(v)
        return dists

    def __len__(self):
        return len(self._probs)

    def depends_on(self):
        """Gets distribution this one depends on.

        Returns
        -------
        dist : Distribution
            Distribution this one depends on. None if the distribution is
            independent.
        """
        if isinstance(self._probs, Distribution):
            return self._probs
        else:
            return None

    @staticmethod
    def is_discrete(values, size):
        """Checks if the distribution is discrete.

        Parameters
        ----------
        values : array_like
            List of variable values.
        size : int
            The length of intensity matrix along variable dimension.

        Returns
        -------
        result : bool
            True if variable is discrete, i.e. len(values) == size.
        """
        if len(values) == size:
            return True
        elif len(values) == size + 1:
            return False
        else:
            raise ValueError("Inconsistent size of values.")


def create_bin_distributions(bins, start_name=1):
    """Creates bin distributions for specified bins.

    Parameters
    ----------
    bins : array_like
        A list of bin boundaries.
    start_name : int
        Starting name for distributions. For every new distribution the name
        is incremented by 1.

    Returns
    -------
    free_name : int
        Distribution name that can be used for new distributions.
    distributions : list
        A list of distributions created. Index in the list corresponds to the
        index of bin.
    """
    distributions = []
    for low, high in zip(bins[:-1], bins[1:]):
        distributions.append(Distribution(start_name, [low, high], [1]))
        start_name += 1
    return start_name, distributions


def expand_matrix_distribution(intensities, *var_values, start_name=1):
    """Converts matrix distribution to the len(var_values) linear.

    Parameters
    ----------
    intensities : array_like
        A matrix of source intensities.
    var_values : tuple
        A tuple of variable values along each axis. Length of var_values must
        be equal to the number of intensities dimensions.
    start_name : int
        Starting name for distributions. Default: 1.

    Returns
    -------
    free_name : int
        The name that can be used for new distributions.
    exp_var_values : tuple
        A tuple of lists of values for each source variable.
    exp_intensities : list[float]
        A list of expanded intensities for every hyperbin.
    """
    if len(intensities.shape) != len(var_values):
        raise ValueError("Inconsistent number of variables")
    uniq_values = []
    exp_var_values = []
    for dim, values in zip(intensities.shape, var_values):
        if Distribution.is_discrete(values, dim):
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

    def __init__(self, **variables):
        self._variables = variables

    def mcnp_repr(self):
        """Gets a string representation of corresponding MCNP card."""
        tokens = ["SDEF"]
        cards = []
        extra_cards = []
        for k, v in self._variables.items():
            tokens.append("{0}={1}".format(k, Source._var_repr(k, v)))
            if isinstance(v, Distribution):
                cards.append(v.mcnp_repr())
                for ec in sorted(v.get_inner(), key=lambda x: x.name):
                    extra_cards.append(ec.mcnp_repr())
        cards.insert(0, print_card(separate(tokens)))
        cards.extend(extra_cards)
        return "\n".join(cards)

    @staticmethod
    def _var_repr(key, value):
        if isinstance(value, Distribution):
            dep = value.depends_on()
            result = "D{0}".format(value.name)
            if dep:
                result = "F{0} ".format(dep.variable) + result
        else:
            result = str(value)
        return result
