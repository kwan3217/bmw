import pytest
import numpy as np

from bmw.chapter4 import kepler


def test_kepler():
    #From canonical values in Vallado, example 2-4, p102-103
    r0_cu=np.array([ 0.17738,-0.35784,1.04614])
    v0_cu=np.array([-0.71383, 0.54436,0.30723])
    r1_cu_ref=np.array([-0.6616125, +0.6840739, -0.6206809])
    v1_cu_ref=np.array([ 0.4667380, -0.2424455, -0.7732126])
    dt=2.974674
    (r1_cu,v1_cu)=kepler(r0_cu,v0_cu,dt)
    assert np.allclose(r1_cu,r1_cu_ref) and np.allclose(v1_cu,v1_cu_ref)

