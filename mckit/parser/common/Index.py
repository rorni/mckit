from collections import defaultdict


class Index(defaultdict):
    def __missing__(self, key):
        """
        Calls default factory with the key as argument.
        """
        if self.default_factory:
            return self.default_factory(key)
        else:
            return super().__missing__(key)


class IgnoringIndex(Index):
    def __getitem__(self, *args, **kwargs):
        return None
