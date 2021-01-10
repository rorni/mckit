# This is the encoding swallowing non asccii (neither unicode) symbols happening in MCNP models code
import os

from pathlib import Path

MCNP_ENCODING = "Cp1251"


def make_dirs(*dirs):
    def apply(d: Path):
        d.mkdir(parents=True, exist_ok=True)

    map(apply, dirs)


def get_root_dir(environment_variable_name, default):
    return Path(os.getenv(environment_variable_name, default)).expanduser()


def assert_all_paths_exist(*paths):
    def apply(p: Path):
        assert p.exists(), 'Path "{}" doesn\'t exist'.format(p)

    map(apply, paths)
