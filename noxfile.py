import nox

nox.options.sessions = "tests"  # "lint", "black"


# TODO dvp: the following test fail on python 3.8
#  TestRCC.test_box_test[center0-axis0-0.5-0]
#  On pytest runs this tests on bigfoot machine this test shows unstable
#  result: often it run successfully.
#  Skip 3.8 testing for a while, and find the reason
@nox.session(python=["3.7"])
def tests(session):
    session.run("poetry", "install", external=True)
    session.run("pytest", "--cov")


# See <https://cjolowicz.github.io/posts/hypermodern-python-03-linting>`

locations = "mckit", "tests", "noxfile.py"


@nox.session(python=["3.7"])
def lint(session):
    args = session.posargs or locations
    session.install("flake8", "flake8-black")
    session.run("flake8", *args)


@nox.session(python=["3.7"])
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
