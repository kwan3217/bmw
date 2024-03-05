import pytest
import numpy as np

from bmw.chapter1 import su_to_cu


@pytest.mark.parametrize(
    "x,LL,TT,inverse,expected",
    [
        # An object orbiting Earth has a position of <1131340,-2282343,6672423> m
        # and a speed of <-5643.05,4303.33,2428.79> m/s. Convert this to canonical units
        (np.array([[ 1131340.0],[-2282343.0],[ 6672423.0]]),1, 0,False,np.array([[ 0.17737781],[-0.35783850],[1.04613980]])),
        (np.array([[-5643.05],     [4303.33],   [2428.79]]),1,-1,False,np.array([[-0.71382529],[ 0.54435559],[0.30723310]])),
        # We are going to solve the Kepler problem over a time of 40min=2400s. How many canonical time units?
        (2400,0,1,False,2.9746739),
        # The answer is given to us from on high but in canonical units. What is the answer in SI units?
        (np.array([[-0.6616125],[ 0.6840739],[-0.6206809]]),1, 0,True,np.array([[-4219855.2],[4363117.1],[-3958787.8]])),
        (np.array([[ 0.4667380],[-0.2424455],[-0.7732126]]),1,-1,True,np.array([[3689.7346],[-1916.6203],[-6112.5284]]))
    ]
)
def test_su_to_cu(x,LL,TT,inverse,expected):
    # Earth radius used as distance unit length: m
    re=6378137.0
    # Earth gravitational constant:  m,s
    mu=398600.4415e9
    assert np.allclose(su_to_cu(x,a=re,mu=mu,LL=LL,TT=TT,inverse=inverse),expected)
