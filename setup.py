import os
import re
import sys
from numpy import get_include
from distutils.core import setup, Extension

topdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(topdir + "/eclipse")
import VERSION as ver


srcdir = 'src_c/'  # C-code source folder
libdir = 'src_c/'  # Where the shared objects are put

files = os.listdir(srcdir)
# This will filter the results for just the c files:
files = list(filter(lambda x:     re.search('.+[.]c$',     x), files))
files = list(filter(lambda x: not re.search('[.#].+[.]c$', x), files))

inc = [get_include(), libdir]
eca = ['-ffast-math']
ela = [             ]

extensions = []
for i in range(len(files)):
  e = Extension(files[i].rstrip(".c"),
                sources=["{:s}{:s}".format(srcdir, files[i])],
                include_dirs=inc,
                extra_compile_args=eca,
                extra_link_args=ela)
  extensions.append(e)


setup(name         = "eclipse",
      version      = "{:d}.{:d}.{:d}".format(ver.ecl_VER, ver.ecl_MIN,
                                             ver.ecl_REV),
      author       = "Patricio Cubillos",
      author_email = "patricio.cubillos@oeaw.ac.at",
      url          = "https://github.com/pcubillos/eclipse",
      packages     = ["eclipse"],
      license      = ["MIT"],
      description  = "Eclipse light-curve models for the times of the Webb.",
      include_dirs = inc,
      #scripts      = ['ecl.py'],
      #entry_points={"console_scripts": ['foo = MCcubed.mccubed:main']},
      ext_modules  = extensions)
