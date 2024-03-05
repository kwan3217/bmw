"""
Algorithms for solving the Kepler and Gauss problems. the two fundamental problems in
two-body mechanics

The <i>Kepler Problem</i>, or prediction problem, is as follows: Given the GM of the
central body, and the position and velocity of a test particle of negligible mass at 
one time, find the position and velocity at any other time.

The <i>Gauss Problem</i>, or targeting problem, is as follows: Given the GM of the 
central body, the position at which a test particle is now, the target position where
it will be, and the time between the postions, find the velocity of the particle at 
both points such that it will travel from the initial position to the target position
in the given time.

This basically follows the Universal Variable formation found in Bate, Muller, and White
chapters 4 and 5.

These algorithms make use of canonical units. Canonical units are distance and time units
relating to a particular central body, such that the GM of that body is 1. In canonical 
units, an object in a circular orbit of radius one Distance Unit (DU) has a speed of
one DU per Time Unit (TU), and therefore an angular velocity of one radian per TU. The 
length of a DU is arbitrary, but is customarily the radius of the central body for planets
and moons, and 1 AU for the Sun.   
"""

import numpy as np
from collections import namedtuple
import math

from .chapter1 import su_to_cu
from .chapter2 import elorb,Elorb
from .chapter4 import kepler
from .chapter5 import gauss

def herrick_gibbs(rr1,rr2,rr3,t1,t2,t3,mu=1):
    """
    Given three closely-spaced position observations, calculate the
    velocity at the middle observation. From Vallado p444, algorithm 52

    This is much less computationally intensive than the Gauss method,
    and has a better chance of numerical stability.

    :param rr1: Position vector at t1
    :param rr2: Position vector at t2
    :param rr3: Position vector at t3
    :param t1:  Time of first position vector
    :param t2:  Time of second position vector
    :param t3:  Time of third position vector
    :param mu:  Gravitational parameter
    :return:    Velocity at t2
    """
    dt31=t3-t1
    dt32=t3-t2
    dt21=t2-t1
    r1=np.linalg.norm(rr1)
    r2=np.linalg.norm(rr2)
    r3=np.linalg.norm(rr3)
    #Coplanarity (not strictly needed)
    Z23=np.cross(rr2,rr3)
    alpha_cop=np.pi/2-np.arccos(np.dot(Z23,rr1)/(np.linalg.norm(Z23)*np.linalg.norm(rr1)))
    #position spread (not strictly needed)
    cosalpha12=np.dot(rr1,rr2)/(r1*r2)
    cosalpha23=np.dot(rr2,rr3)/(r2*r3)
    vv2=      -dt32 *(1/(dt21*dt31)+mu/(12*r1**3))*rr1+\
         (dt32-dt21)*(1/(dt21*dt32)+mu/(12*r2**3))*rr2+\
               dt21 *(1/(dt32*dt31)+mu/(12*r3**3))*rr3
    return vv2

"""
def test_herrick_gibbs():
    rr1=np.array([3419.85564,6019.82602,2784.60022])
    rr2=np.array([2935.91195,6326.18324,2660.59584])
    rr3=np.array([2434.95205,6597.38674,2521.52311])
    t1=0
    t2=1*60+16.48
    t3=2*60+33.04
    re=6378.1363
    mu=398600.4415
    rr1_cu=su_to_cu(rr1,re,mu,1, 0)
    rr2_cu=su_to_cu(rr2,re,mu,1, 0)
    rr3_cu=su_to_cu(rr3,re,mu,1, 0)
    t1_cu =su_to_cu(t1 ,re,mu,0, 1)
    t2_cu =su_to_cu(t2 ,re,mu,0, 1)
    t3_cu =su_to_cu(t3 ,re,mu,0, 1)
    vv2_cu=herrick_gibbs(rr1_cu,rr2_cu,rr3_cu,t1_cu,t2_cu,t3_cu)
    vv2=su_to_cu(vv2_cu,re,mu,1,-1,inverse=True)
    print(vv2)
"""

if __name__=="__main__":
    #test_herrick_gibbs()
    #test_kepler()
    test_gauss1()
    test_gauss2()
    #print(elorb(np.array([1,0,0]),np.array([0,0,1])))

