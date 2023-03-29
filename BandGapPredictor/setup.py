from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Python library for Molecule Bandgap Prediction'
LONG_DESCRIPTION = 'Python library for Molecule Bandgap Prediction using 3d boxel aproach'

# Setting up
setup(
    name="BandGapPredictor",
    version=VERSION,
    author="Franz Rivera Tellez",    
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['plotly',  'torch','matplotlib','numpy' ,'pybel','torchmetrics','pytorch_lightning','path' ],
    keywords=['python', 'video', 'stream', 'video stream', 'camera stream', 'sockets'],
    
)