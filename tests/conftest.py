# tests/conftest.py
from _pytest.config import Config


# TODO dvp: the following doesn't work, but saved as an example of a session fixture

# import os.path
# import sys
# from pathlib import Path
#
# import pytest
#
# PREFIX_PATH = Path(sys.prefix)
#
#
# @pytest.fixture(scope="session", autouse=True)
# def set_ld_library_path(request):
#     print("Setting LD_LIBRARY_PATH")
#     old_ld_library_path = os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep)
#     lib_path = PREFIX_PATH / "lib"
#     assert lib_path.is_dir()
#     new_ld_library_path = str(lib_path.resolve())
#
#     if new_ld_library_path not in old_ld_library_path:
#
#         if old_ld_library_path:
#             new_ld_library_path = os.path.pathsep.join([new_ld_library_path, *old_ld_library_path])
#             os.environ["LD_LIBRARY_PATH"] = new_ld_library_path
#
#             def finalizer():
#                 os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(old_ld_library_path)
#         else:
#             def finalizer():
#                 pass
#
#         request.addfinalizer(finalizer)


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")
    config.addinivalue_line(
        "markers", 'slow: marks tests as slow (deselect with -m "not slow"'
    )
