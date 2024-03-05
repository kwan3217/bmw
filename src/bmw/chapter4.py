"""
Bate, Mueller, White chapter 4 -- Propagation of state vector (Kepler problem)

The <i>Kepler Problem</i>, or prediction problem, is as follows: Given the GM of the
central body, and the position and velocity of a test particle of negligible mass at
one time, find the position and velocity at any other time.

"""
import numpy as np
from kwanmath.vector import vlength, vdot, vcross

from bmw import su_to_cu, elorb


def CC(z):
    """
    Calculate the Universal Variable C(z) function
    """
    if z>0:
        return (1-np.cos(np.sqrt(z)))/z
    elif z==0:
        return 0.5
    else:
        return (1-np.cosh(np.sqrt(-z)))/z


def SS(z):
    """
    Calculate the Universal Variable S(z) function
    """
    if z>0:
        sz=np.sqrt(z)
        return (sz-np.sin(sz))/sz**3
    elif z==0:
        return 1.0/6.0
    else:
        sz=np.sqrt(-z)
        return (np.sinh(sz)-sz)/np.sqrt((-z)**3)


def kepler(rv0:np.ndarray, vv0:np.ndarray, t:float,
           l_DU:float=None, mu:float=None, eps:float=1e-9)->tuple[np.ndarray,np.ndarray]:
    """
    Given a state vector and time interval, calculate the state after the time interval elapses

    :param rv0:  position vector relative to central body
    :param vv0:  velocity vector relative to central body
    :param t:    time interval from given state to requested state - may be negative
    :param l_DU: optional - used to convert rv0 and vv0 to canonical units. Length of distance canonical
                 unit in standard units implied by rv0. If not set, input state and time are already
                 presumed to be in canonical units
    :param mu:   optional, but required if l_DU= is set - used to convert to canonical units. Gravitational
                 constant in standard units
    :param eps:  optional - loop termination criterion, default to 1e-9
    :return: tuple:
         rv_t - Position vector after passage of time t, in same units as rv0
         vv_t - Velocity vector after passage of time t, in same units as vv0
    """
    if t == 0.0:
        # Shortcut if we ask for zero time interval
        return rv0, vv0
    tau = np.pi * 2.0
    if l_DU is not None:
        rv0 = su_to_cu(rv0, l_DU, mu, 1, 0)
        vv0 = su_to_cu(vv0, l_DU, mu, 1, -1)
        t = su_to_cu(t, l_DU, mu, 0, 1)

    r0 = vlength(rv0)
    v0 = vlength(vv0)

    r0dv0 = vdot(rv0, vv0)

    # Determine specific energy, and thereby the orbit shape
    # These are dependent on the initial state and therefore scalar
    E = v0 ** 2 / 2 - 1 / r0
    alpha = -2 * E  # alpha is 1/a, reciprocal of semimajor axis (in case of parabola. a can be infinite, but never zero

    # Starting guess for x
    if alpha > 0:
        # elliptical
        x0 = t * alpha
    elif alpha == 0:
        # parabolic (this will never really happen)
        hv = vcross(rv0, vv0)
        p = vdot(hv, hv)
        # acot(x)=tau/4-atan(x)
        s = (tau / 4.0 - np.atan(3.0 * t * np.sqrt(1.0 / p ** 2))) / 2.0
        w = np.atan(np.tan(s) ** (1.0 / 3.0))
        x0 = np.sqrt(p) / np.tan(2 * w)  # cot(x)=1/tan(x)
    else:
        # hyperbolic
        sa = np.sqrt(-1.0 / alpha)
        st = 1 if t > 0 else -1
        x0_a = st * sa
        x0_n = -2 * alpha * t
        x0_d = r0dv0 + st * sa * (1 - r0 * alpha)
        x0 = x0_a * np.log(x0_n / x0_d)

    done = False
    xn = x0
    while not done:
        z = xn ** 2 * alpha
        C = CC(z)
        S = SS(z)
        r = xn ** 2 * C + r0dv0 * xn * (1 - z * S) + r0 * (1.0 - z * C)
        tn = xn ** 3 * S + r0dv0 * xn ** 2 * C + r0 * xn * (1.0 - z * S)
        xnp1 = xn + (t - tn) / r
        done = np.abs(xn - xnp1) < eps
        xn = xnp1
    x = xn
    f = 1 - x ** 2 * C / r0
    g = t - x ** 3 * S
    fdot = x * (z * S - 1) / (r * r0)
    gdot = 1 - x ** 2 * C / r
    rv_t = f * rv0 + g * vv0
    vv_t = fdot * rv0 + gdot * vv0
    if l_DU is not None:
        rv_t = su_to_cu(rv_t, l_DU, mu, 1, 0, inverse=True)
        vv_t = su_to_cu(vv_t, l_DU, mu, 1, -1, inverse=True)
    return rv_t, vv_t


def plot_kepler():
    import matplotlib.pyplot as plt
    r0_cu=np.array([ 0.17738,-0.35784,1.04614])
    v0_cu=np.array([-0.71383, 0.54436,0.30723])
    print("plot_kepler elorb: ",elorb(r0_cu, v0_cu))
    r_cu=np.zeros([50,3])
    for i,dt in enumerate(np.linspace(0,2.974674)):
        (r1_cu,v1_cu)=kepler(r0_cu,v0_cu,dt)
        r_cu[i,:]=r1_cu
    plt.subplot(221)
    plt.plot(r_cu[:,0],r_cu[:,1])
    plt.subplot(222)
    plt.plot(r_cu[:,0],r_cu[:,2])
    plt.subplot(223)
    plt.plot(r_cu[:,1],r_cu[:,2])
    plt.show()

