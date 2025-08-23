###############################
# Imports # Imports # Imports #
###############################

import os
import shutil
from pathlib import Path
from argparse import Namespace

###################################
# Initialization # Initialization #
###################################

def init():
    """
    Function called once to initialize the Config class

    Note: SERIOUSLY! Call this only once every time you run something. Every time it's initialized it gets re-written from scratch.
    """
    global config
    config = Config()

###############################
# Config Class # Config Class #
###############################

class Config(Namespace):
    """
    Config class helps manage all the different directories and hyperparameters used in this repo. Uses a `Namespace` internally to allow accessing variables (ex. `a`) as `<Config>.a` or `<Config>['a']. Implements dict functions.

    Official docs as a reference: https://docs.python.org/3/reference/datamodel.html

    Initialize once to instantiate the global variable.
    """
    def __init__(self) -> None:
        # Locate parent directory
        self.top_dir = str(Path(__file__).parent.parent.absolute())

        # Package name
        self.package_name = "dataset_dynamics"

        # Initialize normal values by defualt
        self.init_normal()

        return None

    def init_normal(self) -> None:
        """
        Initialize normal configuration values

        Returns: `None`
        """

        ###################################################
        # Dataset File Structure # Dataset File Structure # 
        ###################################################

        # Directory structure
        # - Inputs
        self.dataset_dir = os.path.join(self.top_dir, "Datasets")

        # - Outputs
        self.update_top_dir(self.top_dir)

        self.yalefaces_dir = os.path.join(self.dataset_dir, "yalefaces")

        self.device = 'cpu'

        return None

    def update_top_dir(self, new_top_dir):
        """
        Change the top-level directory

        Returns: `None`
        """
        self.top_dir = new_top_dir

        self.mp4_dir = os.path.join(self.top_dir, "mp4")
        self.jpg_dir = os.path.join(self.top_dir, "jpg")
        self.jpg_mass_dir = os.path.join(self.top_dir, "jpg", "mass")
        self.pdf_dir = os.path.join(self.top_dir, "pdf")
        self.pdf_mass_dir = os.path.join(self.top_dir, "pdf", "mass")
        self.pkl_dir = os.path.join(self.top_dir, "pkl")

        return None

    def clear_top_dir(self):
        """
        Delete top-level directory and re-creates it to be empty
        
        Returns: `None`
        """
        # Remove if it exists
        if os.path.exists(config.top_dir):
            shutil.rmtree(config.top_dir)

        # Create it again
        if not os.path.exists(config.top_dir):
            os.makedirs(os.path.join(config.top_dir))

        return None

    def create_output_dirs(self):
        """
        Create output directories

        Returns: `None`
        """
        # Create those directories
        for newpath in [self.top_dir, self.mp4_dir, self.jpg_dir, self.pdf_dir, self.pkl_dir, self.pdf_mass_dir, self.jpg_mass_dir]:
            if not os.path.exists(newpath):
                os.makedirs(newpath)
        
        return None

    ########################
    # Dict-based Functions #
    ########################

    # Source: https://stackoverflow.com/questions/4014621/a-python-class-that-acts-like-dict
    # These have been modified to update the internal Namespace

    def __setitem__(self, key, item):
        vars(self)[key] = item

    def __getitem__(self, key):
        return vars(self)[key]

    def __repr__(self):
        return repr(vars(self))

    def __len__(self):
        return len(vars(self))

    def __delitem__(self, key):
        del vars(self)[key]

    def clear(self):
        return vars(self).clear()

    def copy(self):
        return vars(self).copy()

    def has_key(self, k):
        return k in vars(self)

    def update(self, *args, **kwargs):
        return vars(self).update(*args, **kwargs)

    def keys(self):
        return vars(self).keys()

    def values(self):
        return vars(self).values()

    def items(self):
        return vars(self).items()

    def pop(self, *args):
        return vars(self).pop(*args)

    def __iter__(self):
        return iter(vars(self))

    def __str__(self):
        return str(repr(vars(self)))