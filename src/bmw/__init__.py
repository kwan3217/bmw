"""
Algorithms for solving the Kepler and Gauss problems. the two fundamental problems in
two-body mechanics

The code is divided into chapters, one file per chapter. Each chapter has the table of
contents from the book, along with a tick mark for the parts that are implemented
and the function name(s) that implement it.

"""

from .chapter1 import su_to_cu
from .chapter2 import elorb,Elorb,herrick_gibbs
from .chapter4 import kepler
from .chapter5 import gauss


