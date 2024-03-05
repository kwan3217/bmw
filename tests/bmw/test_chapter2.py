import pytest
import numpy as np

from bmw.chapter2 import elorb,Elorb


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


