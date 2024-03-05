import pytest
import numpy as np

from bmw import su_to_cu
from bmw.chapter2 import elorb, Elorb, herrick_gibbs


@pytest.mark.parametrize(
    "rv,vv,ref_elorb",
    [
        (np.array([[2.0],[0.0],[0.0]]),np.array([[0.0],[1.0],[0.0]]),
         Elorb(p=4.0, a=np.inf, e=1.0, i=0.0, an=np.nan, ap=np.nan,
               ta=0.0, tp=-0.0, rp=2.0, MM=0.0, n=0.1767766952966369, t=np.inf))
    ]
)
def test_elorb(rv,vv,ref_elorb):
    this_elorb=elorb(rv,vv)
    assert np.allclose(this_elorb,ref_elorb,equal_nan=True)


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


