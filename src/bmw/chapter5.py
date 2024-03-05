"""
Bate, Mueller, White chapter 5 -- Solve the targeting problem.
In BMW this is called the Gauss problem, and in other contexts
this is called the Lambert problem.
"""
import numpy as np
from kwanmath.vector import vangle, vlength, vdot, vcross

from bmw import su_to_cu
from bmw.chapter4 import SS, CC


def gauss(rv1, rv2, t, Type=-1, l_DU=None, mu=None, eps=1e-9):
    def FindTTrialCore(A, S, X, Y):
        return (X ** 3) * S + A * np.sqrt(Y)

    def FindTTrial(A, r1, r2, Z):
        S = SS(Z)
        C = CC(Z)
        if C == 0:
            return float('inf')
        Y = r1 + r2 - A * (1 - Z * S) / np.sqrt(C)
        X = np.sqrt(Y / C)
        return FindTTrialCore(A, S, X, Y)

    def FindZLo(A, r1, r2):
        eps = 1e-9
        # Find the Z which results in a Y of exactly zero, by bisection
        Zhi = 0.0;
        Y = 1.0;
        Zlo = -1.0;

        while Y > 0:
            Zlo *= 2.0;
            Y = r1 + r2 - A * (1 - Zlo * SS(Zlo)) / np.sqrt(CC(Zlo))

        while True:  # Emulate a repeat/until loop
            Z = (Zlo + Zhi) / 2
            Y = r1 + r2 - A * (1 - Z * SS(Z)) / np.sqrt(CC(Z))
            if Y * Zlo > 0:
                Zlo = Z
            else:
                Zhi = Z
            if np.abs(Zlo - Zhi) < eps:  # repeat until this condition is true
                break
        return Z + 1e-5

    def FindZLo2(A, r1, r2, T):
        # Find the Z which results in a TTrial of less than T
        Z = -1.0
        TTrial = FindTTrial(A, r1, r2, Z) - T
        while TTrial > 0:
            Z *= 2
            TTrial = FindTTrial(A, r1, r2, Z) - T
        return Z

    tau = 2 * np.pi
    if l_DU is not None:
        rv1 = su_to_cu(rv1, l_DU, mu, 1, 0)
        rv2 = su_to_cu(rv2, l_DU, mu, 1, 0)
        t = su_to_cu(t, l_DU, mu, 0, 1)

    if (Type < 0):
        pole = vcross(rv1, rv2)
        if (pole[2]) > 0:
            # prograde is short way
            Type = (-Type - 1) * 2 + 1
        else:
            # prograde is long way way
            Type = (-Type - 1) * 2 + 2

    if t < 0:
        raise ValueError("Time to intercept is negative. Time travel is not allowed in this universe!")
    r1 = vlength(rv1)
    r2 = vlength(rv2)
    r1dr2 = vdot(rv1, rv2)
    DeltaNu = vangle(rv1, rv2)
    Revs = (Type - 1) / 2;
    # short-way and long-way are reversed for odd-numbers of complete revs
    if ((Revs % 2) == 1) ^ ((Type % 2) == 1):
        # Short way
        DM = 1.0
    else:
        # Long way
        DM = -1.0
        DeltaNu = tau - DeltaNu
    if Revs > 0:
        minA = r1 / 2.0
        minT = Revs * np.sqrt(tau * minA ** 3)
        if minT > t:
            raise ValueError(
                "Can't do it! Minimum trip time for %d revs is %fTU, more than requested %fTU" % (Revs, minT, t))
    A = DM * np.sqrt(r1 * r2 * (1 + np.cos(DeltaNu)))
    if (Revs < 1):
        # less than one rev
        if Type == 1:
            Zlo = FindZLo(A, r1, r2)
        else:
            Zlo = FindZLo2(A, r1, r2, t)
        Zhi = tau ** 2;
    else:
        # more than one rev
        # Use Zeno's method
        Zlo = ((2 * Revs + 1) * tau / 2.0) ^ 2  # Z that gives the lowest TIME, not necessarily lowest Z
        # Zbound is the value of Z which gives an infinite T
        if (Type % 2) == 1:
            Zbound = (Revs * tau) ** 2
        else:
            Zbound = ((Revs + 1) * tau) ** 2
        Zhi = (Zbound + Zlo) / 2.0  # Z that gives the highest TIME, not necessarily highest Z
        while True:  # emulate repeat/until
            Thi = FindTTrial(A, r1, r2, Zhi)
            Zhi = (Zbound + Zhi) / 2;  # Split the difference between current Zhi and bound
            if Thi >= t:
                break

    # Solve it by bisection
    tnlo = FindTTrial(A, r1, r2, Zlo)
    tnhi = FindTTrial(A, r1, r2, Zhi)
    while True:  # emulate repeat/until
        Z = (Zlo + Zhi) / 2.0
        tn = FindTTrial(A, r1, r2, Z)
        if (t - tn) * tnlo > 0:
            Zlo = Z
        else:
            Zhi = Z
        if abs(Zlo - Zhi) <= eps:
            break

    S = (SS(Z))
    C = (CC(Z))
    Y = r1 + r2 - A * (1.0 - Z * S) / np.sqrt(C)
    f = 1.0 - Y / r1
    g = A * np.sqrt(Y)
    gdot = 1.0 - Y / r2
    vv1 = (rv2 - rv1 * f) / g
    vv2 = (rv2 * gdot - rv1) / g
    if l_DU is not None:
        vv1 = su_to_cu(vv1, l_DU, mu, 1, -1, inverse=True)
        vv2 = su_to_cu(vv2, l_DU, mu, 1, -1, inverse=True)
    return (vv1, vv2)

