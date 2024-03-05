"""
Algorithms for solving the Kepler and Gauss problems. the two fundamental problems in
two-body mechanics

"""

import numpy as np
from collections import namedtuple
import math

from .chapter1 import su_to_cu
from .chapter2 import elorb,Elorb,herrick_gibbs
from .chapter4 import kepler
from .chapter5 import gauss


