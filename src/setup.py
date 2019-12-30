# Don't run this directly, use the build.sh script

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

import shutil
import os
import sys

def findFilesWithExtension(ext):
    result = []
    def recurseGather(curDir):
        nonlocal result;
        names = list(os.listdir(curDir))        
        for name in names:
            fpath = os.path.join(curDir, name)
            if os.path.isfile(fpath) and fpath.endswith(ext):
                result.append(fpath)
            elif os.path.isdir(fpath):
                recurseGather(fpath)
    recurseGather(".")
    return result

pyxFiles = findFilesWithExtension(".pyx")


if "clean" in sys.argv:
    def replaceExtAndDelete(f, oldExt, newExt):
        newName = f.replace(oldExt, newExt)
        if os.path.isfile(newName):
            os.remove(newName)

    shutil.rmtree("build", True)

    # delete all *.so, *.html, *.c files that were created by previous builds
    soFiles = findFilesWithExtension(".so");
    for soFile in soFiles:
        os.remove(soFile)

    for pyxFile in pyxFiles:
        replaceExtAndDelete(pyxFile, ".pyx", ".html")
        replaceExtAndDelete(pyxFile, ".pyx", ".c")
else:
    print("Found pyx files:")

    def makeModulePath(pyxFile):
        return pyxFile.replace("./", "").replace(".pyx", "").replace("/", ".")

    for pyxFile in pyxFiles:
        print(makeModulePath(pyxFile), pyxFile)

    myExtensions = list(map(lambda f: Extension(makeModulePath(f), [f]), pyxFiles))

    setup(
        ext_modules=cythonize(myExtensions, annotate=True, compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'nonecheck': False,
            'language_level': "3"
        }),
        include_dirs=[numpy.get_include()]
    )
