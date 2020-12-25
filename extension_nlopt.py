import re
from pathlib import Path
from subprocess import check_call

from setuptools import Extension


def get_nlopt_version(source_dir: Path) -> str:
    check_call(
        "git submodule update --init --recursive --depth=1".split(), cwd=Path.cwd()
    )
    with open(source_dir / "CMakeLists.txt") as f:
        content = f.read()
        version = []
        for s in ("MAJOR", "MINOR", "BUGFIX"):
            m = re.search(f"NLOPT_{s}_VERSION *['\"](.+)['\"]", content)
            version.append(m.group(1))
        version = ".".join(version)
        return version


class NLOptBuildExtension(Extension):
    def __init__(self, name: str = "nlopt"):
        super().__init__(name, sources=[])
        # Source dir should be at the root directory
        self.source_dir = Path(__file__).parent.absolute() / "3rd-party" / "nlopt"
        self.version = get_nlopt_version(self.source_dir)
        self.language = "c++"
