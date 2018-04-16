# Copyright (c) 2018 Patricio Cubillos and contributors.
# eclipse is open-source software under the MIT license (see LICENSE).

# Commands:
from .eclmodels import *
from . import VERSION as ver

__all__ = eclmodels.__all__
__version__ = "{:d}.{:d}.{:d}".format(ver.ECL_VER, ver.ECL_MIN, ver.ECL_REV)

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__ ):
        del locals()[varname]
del(varname)
