import os, sys

def mrcnnPath():
    filePath = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(filePath, os.pardir, os.pardir))

def currentFilePath(file=None):
    file = file if file else __file__
    return os.path.dirname(os.path.realpath(file))

def mrcnnToPath():
    sys.path.append(mrcnnPath())