# read version from installed package
from importlib.metadata import version
__version__ = version("scmrelax")

# populate package namespace
from scmrelax.scmrelax import fit