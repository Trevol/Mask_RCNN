import os, sys


ROOT_DIR = os.path.abspath("../../../..")
sys.path.append(ROOT_DIR)
from mrcnn.model import MaskRCNN

def findImportRoot():
    # echo $VIRTUAL_ENV
    # starting from current dir find dir containing venv
    pass