"""
Bate, Mueller, White chapter 1

TWO-BODY ORBITAL MECHANICS
[ ] 1.1 Historical Background and Basic Laws
[ ] 1.2 The N-Body Problem
[ ] 1.3 The Two-Body Problem
[ ] 1.4 Constants of the Motion
[ ] 1.5 The Trajectory Equation
[ ] 1.6 Relating \script{E} and $h$ to the Geometry of an Orbit
[ ] 1.7 The Elliptical Orbit
[ ] 1.8 The Circular Orbit
[ ] 1.9 The Parabolic Orbit
[ ] 1.10 The Hyperbolic Orbit
[/] 1.11 Canonical Units
  * su_to_cu()

These algorithms make use of canonical units. Canonical units are distance and time units
relating to a particular central body, such that the GM of that body is 1. In canonical
units, an object in a circular orbit of radius one Distance Unit (DU) has a speed of
one DU per Time Unit (TU), and therefore an angular velocity of one radian per TU. The
length of a DU is arbitrary, but is customarily the radius of the central body for planets
and moons, and 1 AU for the Sun.
"""

import numpy as np

def su_to_cu(x:float|np.ndarray,
             a:float,
             mu:float,
             LL:float|int,
             TT:float|int,
             inverse:bool=False):
    """
    Convert standard units to canonical units

    :param x:  measurement to convert, may be scalar or any size or shape of vector
    :param a:  length of canonical distance unit in standard units. Standard length unit is specified by this.
    :param mu: gravitational constant in standard distance and time units. Standard length unit same as
               above, standard time unit implied by this.
    :param LL: power of length dimension used in x
    :param TT: power of time dimension used in x
    :param inverse: - convert x from canonical units back to standard units
    :return: a scalar or array the same size as x, in canonical units (or standard if /inv was set)

    To convert:  Standard Unit to       Canonical Unit          Multiply by
    Distance     m                      DU                      1/a
    Time         s                      TU                      sqrt(mu/a**3)
    For derived units, raise the base unit for each dimension to the power of the dimension needed, then multiply.
     For example
    Speed        m/s                    DU/TU (LL=1,TT=-1)      1/a*sqrt(a**3/mu)
    For inverse conversion, divide instead of multiply
    """
    DU=(1.0/a)**LL
    TU=np.sqrt(mu/a**3)**TT
    DUTU=DU*TU
    if inverse:
        DUTU=1.0/DUTU
    return x*DUTU

