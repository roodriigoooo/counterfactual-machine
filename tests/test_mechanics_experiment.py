import torch
import numpy as np
from counterfactual.contact_mechanics import MechanicsModel
from counterfactual.contact_suite import fixtures
from counterfactual.recurrent import pack


def test_revised_impact_responds_and_miss_stays_free():
    model=MechanicsModel('swept','local').eval()
    scene=fixtures()[3];state,context=pack(scene)
    state[:2]=[3.,2.45];state[2:4]=[0.,-3.]
    x=torch.tensor(state[None]);c=torch.tensor(context[None])
    y,_=model(x,c,1)
    assert y[0,1,3]>x[0,3]  # suppress-all-interactions cannot pass
    y[0,1,3].backward()
    assert model.response.weight.grad[1].abs().sum()>0


def test_memory_free_control_is_trained_capacity_not_memory_reset():
    local=MechanicsModel('swept','local');recurrent=MechanicsModel('swept','gru')
    ratio=sum(p.numel() for p in local.parameters())/sum(p.numel() for p in recurrent.parameters())
    assert .9<ratio<1.1
    edge=torch.randn(2,32)
    torch.testing.assert_close(local.memory(edge,torch.zeros_like(edge)),local.memory(edge,torch.ones_like(edge)))


def test_swept_permutation_and_persistent_zero_edit():
    model=MechanicsModel('swept').eval();a,c=pack(fixtures()[5]);order=[4,1,3,0,2]
    a=torch.tensor(a[None]);c=torch.tensor(c[None])
    with torch.no_grad():
        x,s=model(a,c);y,t=model(a,c[:,order])
    torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(s[:,:,order],t,atol=1e-5,rtol=1e-5)
