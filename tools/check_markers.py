#!python3
"""Show markers that can be used on Github Actions."""

import sys

from pathlib import Path

import poetry.utils.env as env

se = env.SystemEnv(Path(sys.prefix))
print("markers:\n", se.marker_env)
print("paths:\n", se.paths)
