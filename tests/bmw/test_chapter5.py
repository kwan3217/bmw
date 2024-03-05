import pytest
import numpy as np
from kwanmath.vector import vcomp

from bmw import gauss

@pytest.mark.parametrize(
    "r0_cu,v0_cu_ref,r1_cu,v1_cu_ref,dt,Type",
    [
        (vcomp((2.5,0.0,0.0)),vcomp((0.2604450,0.3688589,0.0)),
         vcomp((1.915111, 1.606969, 0.0)),vcomp((-0.4366104, 0.1151515, 0.0)),5.6519,-1),
        (vcomp((0.17738, -0.35784, 1.04614)),vcomp((-0.71383, 0.54436, 0.30723)),
         vcomp((-0.6616125, +0.6840739, -0.6206809)),vcomp((0.4667380, -0.2424455, -0.7732126)),2.974674,1),
    ]
)
def test_gauss(r0_cu,v0_cu_ref,r1_cu,v1_cu_ref,dt,Type):
    #Canonical unit numbers from Vallado example 7-5, p467
    #and time between, and find velocities
    (v0_cu,v1_cu)=gauss(r0_cu,r1_cu,dt,Type=Type)
    # Loosten up limits a little bit since the book used a different
    # number of iterations than we use.
    delta=np.abs(v0_cu-v0_cu_ref)
    rtol=5e-5
    atol=1e-8
    limit=atol+rtol*v0_cu_ref
    assert np.allclose(v0_cu,v0_cu_ref,rtol=rtol,atol=atol)
    delta=np.abs(v1_cu-v1_cu_ref)
    limit=atol+rtol*v1_cu_ref
    assert np.allclose(v1_cu,v1_cu_ref,rtol=rtol,atol=atol)
