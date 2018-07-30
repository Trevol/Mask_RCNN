#$env:PYTHONPATH = Get-Location
import os
import mrcnn.model
import pathUtils

from ContactsDataset import ContactsDataset
from ContactsConfig import ContactsConfig

dataset = ContactsDataset()
config = ContactsConfig()
model = mrcnn.model.MaskRCNN("training", config, os.path.join(pathUtils.currentFilePath(__file__), 'logs'))

