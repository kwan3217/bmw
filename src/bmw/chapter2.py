"""
Bate, Mueller, White chapter 2 -- Orbital elements from state vector
"""
import math
from collections import namedtuple

import numpy as np
from kwanmath.vector import vlength, vcross, vangle, vcomp, vdot, vdecomp

from .chapter1 import su_to_cu

Elorb = namedtuple('elorb', ['p', 'a', 'e', 'i', 'an', 'ap', 'ta', 'tp', 'rp', 'MM', 'n', 't'])

def elorb(rv:np.ndarray, vv:np.ndarray, l_DU:float=None, mu:float=None, t0:float=None)->Elorb:
    """
    Given state vector, calculate orbital elements.

    :param rv: position vector, in distance units implied by mu, any inertial frame is fine, but center of attraction
               is assumed to be at the origin of the frame
    :param vv: inertial velocity vector, distance and time units implied by mu, must be in same frame as rv
    :param l_DU: length of a distance unit, used for conversion to canonical units internally. Default is to assume
                 that the units are already canonical.
    :param mu: Gravity parameter, implies distance and time units. Only used for canonical unit conversion, and therefore
               ignored if l_DU is not passed.
    return
      a named tuple
        p:  semi-parameter, distance from focus to orbit at TA=+-90deg, in original distance units, always positive for any eccentricity
        a:  semimajor axis, in original distance units
        e:  eccentricity
        i:  inclination, radians
        an: longitude of ascending node, angle between x axis and line of intersection between orbit plane and xy plane, radians
        ap: argument of periapse, angle between xy plane and periapse along orbit plane, radians
        ta: true anomaly, angle between periapse and object, radians
        tp: time to next periapse, in original time units. Negative if only one periapse and in the past
        rp: radius of periapse in original distance units
    """
    if l_DU is not None:
        rv = su_to_cu(rv, l_DU, mu, 1, 0)
        vv = su_to_cu(vv, l_DU, mu, 1, -1)
    r = vlength(rv)
    v = vlength(vv)

    hv = vcross(rv, vv)
    h = vlength(hv)
    nv = vcross(vcomp((0, 0, 1)), hv)
    n = vlength(nv)
    ev = (v ** 2 - 1.0 / r) * rv - vdot(rv, vv) * vv
    e = vlength(ev)

    xi = v ** 2 / 2.0 - 1 / r
    if e != 1.0:
        a = -1 / (2 * xi)
        p = a * (1 - e ** 2)
    else:
        # parabolic case
        p = h ** 2
        a = math.inf
    hx, hy, hz = vdecomp(hv)
    nx, ny, nz = vdecomp(nv)
    i = np.arccos(hz / h)
    an = np.arccos(nx / n)
    if ny < 0:
        an = 2 * np.pi - an
    ap = vangle(ev, nv)
    ex, ey, ez = vdecomp(ev)
    if ez < 0:
        ap = 2 * np.pi - ap
    ta = vangle(ev, rv)
    if vdot(rv, vv) < 0:
        ta = 2 * np.pi - ta
    if not np.isfinite(a):
        EE = np.tan(ta) / 2
        MM = EE ** 3 / 3 + EE
        n = np.sqrt(2 / (p ** 3))
        rp = p / 2
    elif a > 0:
        EE = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(ta / 2))
        MM = EE - e * np.sin(EE)
        n = np.sqrt(1 / (a ** 3))
        rp = a * (1 - e)
    elif a < 0:
        EE = np.arcsinh(np.sin(ta) * np.sqrt(e ** 2 - 1) / (1 + e * np.cos(ta)))
        MM = e * np.sinh(EE) - EE
        n = np.sqrt(-1 / (a ** 3))
        rp = a * (1 - e)
    tp = -MM / n
    t = 2 * np.pi * np.sqrt(a ** 3)
    if l_DU is not None:
        p = su_to_cu(p, l_DU, mu, 1, 0, inverse=True)
        a = su_to_cu(a, l_DU, mu, 1, 0, inverse=True)
        tp = su_to_cu(tp, l_DU, mu, 0, 1, inverse=True)
        if t0 is not None:
            tp += t0
        rp = su_to_cu(rp, l_DU, mu, 1, 0, inverse=True)
        n = su_to_cu(n, l_DU, mu, 0, -1, inverse=True)
        t = su_to_cu(t, l_DU, mu, 0, 1, inverse=True)
    return Elorb(p=p, a=a, e=e, i=i, an=an, ap=ap, ta=ta, tp=tp, rp=rp, MM=MM, n=n, t=t)


def herrick_gibbs(rr1:np.ndarray,rr2:np.ndarray,rr3:np.ndarray,
                  t1:float,t2:float,t3:float,mu:float=1):
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
    r1=vlength(rr1)
    r2=vlength(rr2)
    r3=vlength(rr3)
    #Coplanarity (not strictly needed)
    Z23=vcross(rr2,rr3)
    alpha_cop=np.pi/2-vangle(Z23,rr1)
    #position spread (not strictly needed)
    cosalpha12=np.dot(rr1,rr2)/(r1*r2)
    cosalpha23=np.dot(rr2,rr3)/(r2*r3)
    vv2=      -dt32 *(1/(dt21*dt31)+mu/(12*r1**3))*rr1+\
         (dt32-dt21)*(1/(dt21*dt32)+mu/(12*r2**3))*rr2+\
               dt21 *(1/(dt32*dt31)+mu/(12*r3**3))*rr3
    return vv2
