import numpy as np
import pytest
import torch
from counterfactual.recurrent import InteractionRollout, pack
from counterfactual.schemas import SceneSpec, Ball, Bumper, GoalSpec, Region
from counterfactual.scenes import generate_scene
from counterfactual.outcomes import BaselineOutcome
from counterfactual.planning import plan


def test_recurrent_free_flight_and_unreachable_edit_identity():
    model=InteractionRollout().eval()
    scene=SceneSpec(id='flight',parent_id='flight',ball=Ball(position=(.5,3.5),velocity=(0,0)),bumper=Bumper(position=(5,2)))
    a,c=pack(scene)
    shifted=c.copy();shifted[0,1]+=.12;shifted[0,3]+=.12
    with torch.no_grad():
        states,contacts=model(torch.tensor(np.stack([a,a])),torch.tensor(np.stack([c,shifted])))
    torch.testing.assert_close(states[0],states[1],atol=0,rtol=0)
    t=np.arange(121)/60
    np.testing.assert_allclose(states[0,:,1],3.5-9.81*t*(t-1/1920)/2,atol=3e-5)
    assert contacts.sigmoid().max()<1e-8


def test_permutation_and_zero_identity():
    torch.manual_seed(3);model=InteractionRollout().eval()
    a,c=pack(generate_scene(1));p=[3,1,4,0,2]
    a=torch.tensor(a[None]);c=torch.tensor(c[None])
    with torch.no_grad():
        x,s=model(a,c);y,t=model(a,c[:,p])
    torch.testing.assert_close(x,y,atol=1e-4,rtol=1e-4)
    torch.testing.assert_close(s[:,:,p],t,atol=1e-4,rtol=1e-4)


def test_contact_head_drives_motion_and_recomputed_geometry():
    model=InteractionRollout()
    scene=generate_scene(0);a,c=pack(scene)
    a[:2]=np.array(scene.bumper.position)+[0,scene.bumper.radius+scene.ball.radius+.001]
    a[2:4]=[0,-3]
    states,_=model(torch.tensor(a[None]),torch.tensor(c[None]),1)
    states[:,-1,3].sum().backward()
    assert model.response.weight.grad[0].abs().sum()>0
    assert model.edge[0].weight.grad.abs().sum()>0


def test_baseline_guard_binds_evidence_and_separates_known_success():
    scene=generate_scene(0);goal=GoalSpec(region=Region(radius=99))
    class Fake:
        identity={}
        def forecast(self,scene,edits):
            return np.zeros((len(edits),121,6)),np.zeros((len(edits),120,5))
    failure=BaselineOutcome.from_verification(scene,goal,dict(valid=True,success=False,margin=-1))
    p=plan(scene,goal,Fake(),count=3,baseline=failure)
    assert p['grid'][p['selected_index']] != 0
    success=failure.model_copy(update={'success':True,'margin':1})
    p=plan(scene,goal,Fake(),count=3,baseline=success)
    assert p['grid'][p['selected_index']]==0 and p['status']=='known baseline success'
    with pytest.raises(ValueError):
        plan(scene,goal.model_copy(update={'kind':'avoid'}),Fake(),baseline=success)
    with pytest.raises(ValueError):
        plan(scene,goal,Fake(),baseline=success.model_copy(update={'configuration_hash':'stale'}))
