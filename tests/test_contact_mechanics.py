import numpy as np
import torch
from counterfactual.contact_suite import fixtures,match_events
from counterfactual.contact_mechanics import swept_admission,MechanicsModel
from counterfactual.recurrent import pack,DT
from counterfactual.simulator import simulate


def test_admission_rejects_grazing_and_separating_and_admits_impact():
    c=pack(fixtures()[3])[1]
    # Top of circle, moving away vs moving into the same surface.
    states=torch.tensor([[3.,2.45,0.,3.,0.,0.],[3.,2.45,0.,-3.,0.,0.],[3.441,2.05,0.,-3.,0.,0.]])
    hit,t,_,_=swept_admission(states,torch.tensor(np.stack([c]*3)))
    assert not hit[0].any() and hit[1,0] and not hit[2].any()
    assert 0<t[1,0]<DT


def test_revised_no_response_without_admission_and_classifier_not_force():
    model=MechanicsModel('swept').eval();a,c=pack(fixtures()[2]);a[2:4]=[3.,0.]
    a=torch.tensor(a[None]);c=torch.tensor(c[None]);h=torch.zeros(1,5,32)
    with torch.no_grad():
        x,_,_=model.step(a,c,model.encode(c),h)
        model.response.bias[0]=100
        y,_,_=model.step(a,c,model.encode(c),h)
    torch.testing.assert_close(x,y,atol=0,rtol=0)
    expected=a[:,2:4]+a.new_tensor([0,-9.81])*DT
    torch.testing.assert_close(x[:,2:4],expected)


def test_diagnostic_response_capture_does_not_change_physics():
    scene=fixtures()[3];a=simulate(scene);b=simulate(scene,record_responses=True)
    np.testing.assert_array_equal(a.states,b.states)
    assert a.events==b.events and b.responses and not a.responses
    e=b.responses[0]
    dv=np.subtract(e['after'][2:4],e['before'][2:4])-np.array([0,-9.81/1920])
    np.testing.assert_allclose(dv,e['impulse_on_ball'],atol=1e-8)


def test_event_matching_distinguishes_small_shift_from_invention():
    m=match_events([.2,.5],[.21,.9]);assert m['matched']==1 and m['invented']==1 and m['missed']==1
