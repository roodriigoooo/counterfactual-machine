import numpy as np
import torch
from counterfactual.phase_training import phase_objective
from counterfactual.phase_evaluation import phase_metrics


def test_weight_normalization_is_relative_scale_not_double_weighting():
    logits=torch.zeros(3,4);labels=torch.tensor([0,1,3]);mask=torch.ones(3)
    torch.testing.assert_close(phase_objective(logits,labels,mask,'weights'),torch.tensor(4.).log())
    torch.testing.assert_close(phase_objective(logits,labels,mask,'count'),torch.tensor(4.).log()*(.05+1+.5)/3)
    mask[2]=0
    torch.testing.assert_close(phase_objective(logits,labels,mask,'weights'),torch.tensor(4.).log())


def test_mask_cannot_hide_raw_false_positives_or_parent_support():
    probability=np.array([[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0.]])
    truth=np.array([1,0,0,0]);mask=np.ones(4,bool);geometry=np.array([True,False,False,False]);parents=np.array([0,0,1,1])
    m=phase_metrics(probability,truth,mask,geometry,geometry,parents)
    assert m['raw_contact']['fp']==3 and m['masked_contact']['f1']==1
    assert m['always_positive_masked']==m['geometry']
    assert m['phase_support'][0]=={'intervals':3,'parents':2}


def test_callbacks_can_coexist_and_rebegin_during_occupancy():
    from types import SimpleNamespace
    from counterfactual.contact_events import event_labels
    events=[dict(time=.001,phase='begin',objects=['ball','bumper']),dict(time=.009,phase='separate',objects=['ball','bumper']),dict(time=.012,phase='begin',objects=['ball','bumper']),dict(time=.02,phase='separate',objects=['ball','bumper'])]
    tr=SimpleNamespace(contacts=np.array([[1],[1],[0]],np.float32),object_ids=['bumper'],events=events,responses=[])
    r=event_labels(tr)
    assert r['event_labels'][0,0,1:4].tolist()==[1,1,1]
    assert r['event_counts'][0,0].tolist()==[2,1]
    assert r['active_start'][:,0].tolist()==[0,1,0]
    assert len(r['ordered_events'])==4
    assert r['event_labels'][1,0,2]==1 and r['event_labels'][1,0,3]==0


def test_competing_contact_features_are_permutation_equivariant():
    from counterfactual.episode_model import event_geometry,EpisodeModel
    from counterfactual.episode_data import scene_for
    from counterfactual.recurrent import pack
    state,c=pack(scene_for(40006,6));state=torch.tensor(state[None]);c=torch.tensor(c[None]);order=torch.tensor([2,0,4,1,3])
    a,b,_=event_geometry(state,c);x,y,_=event_geometry(state,c[:,order]);torch.testing.assert_close(a[:,order],x);torch.testing.assert_close(b[:,order],y)
    model=EpisodeModel('E2');active=torch.zeros(1,5)
    a,_=model.readout(state,c,active);b,_=model.readout(state,c[:,order],active[:,order]);torch.testing.assert_close(a[:,order],b)


def test_tolerance_never_bridges_invalid_cells_or_truncates_at_bounds():
    from counterfactual.tolerance import select_with_tolerance,successful_components
    grid=np.linspace(-.12,.12,9);margin=np.ones(9);passed=np.ones(9,bool);valid=np.ones(9,bool)
    valid[4]=False
    r=select_with_tolerance(grid,margin,passed,valid,epsilon=.031,gamma=0)
    assert r['index'] is None  # ceil to two steps; neither four-cell component fits
    valid[:]=True;r=select_with_tolerance(grid,margin,passed,valid,epsilon=.031,gamma=0)
    assert r['index']==4 and np.isclose(r['effective_epsilon_m'],.06)
    passed[4]=False
    assert len(successful_components(grid,passed))==2


def test_zero_edit_nominal_success_does_not_short_circuit_tolerance():
    from counterfactual.tolerance import select_with_tolerance
    grid=np.linspace(-1,1,9);passed=np.zeros(9,bool);passed[4]=True
    nominal=select_with_tolerance(grid,np.ones(9),passed,np.ones(9,bool),epsilon=0,gamma=0)
    robust=select_with_tolerance(grid,np.ones(9),passed,np.ones(9,bool),epsilon=.1,gamma=0)
    assert nominal['index']==4 and robust['index'] is None
    # Strict avoid semantics arrive as passed=False, even when the margin is zero.
    assert select_with_tolerance(grid,np.zeros(9),np.zeros(9,bool),np.ones(9,bool),epsilon=0,gamma=0)['index'] is None


def test_overshoot_is_directional_and_zero_is_symmetric():
    from counterfactual.tolerance import neighborhoods
    windows,effective=neighborhoods(np.linspace(-1,1,9),.25,'overshoot')
    np.testing.assert_array_equal(windows[2],[1,2]);np.testing.assert_array_equal(windows[6],[6,7]);np.testing.assert_array_equal(windows[4],[3,4,5])
    assert windows[0] is None and windows[-1] is None


def test_one_subinterval_preserves_the_analytic_control_and_ties_are_symmetric():
    from counterfactual.subinterval import step
    from counterfactual.architecture import conditions,integrate
    from counterfactual.contact_suite import fixtures
    from counterfactual.recurrent import pack,DT
    s,c=pack(fixtures()[3]);s[:4]=[3.,2.45,0.,-3.];s=torch.tensor(s[None]);c=torch.tensor(c[None])
    _,geo=conditions(s,c);coeff=torch.stack([c[...,8],torch.ones_like(c[...,8])],-1)
    original,_=integrate(s,c,geo,coeff);new,_,_,_=step(s,c,DT)
    torch.testing.assert_close(original,new,rtol=0,atol=0)
    duplicate=c.clone();duplicate[:,1]=duplicate[:,0]
    a=step(s,duplicate,DT)[0];order=torch.tensor([1,0,2,3,4]);b=step(s,duplicate[:,order],DT)[0]
    torch.testing.assert_close(a,b)


def test_nominal_verified_zero_survives_a_disabled_failing_task_rule():
    from counterfactual.tolerance_experiment import choose
    grid=np.linspace(-.12,.12,9);valid=np.ones(9,bool);table={'margins':np.ones((1,9)),'passed':np.ones((1,9),bool)}
    base={'success':True,'margin':.02}
    assert choose(None,grid,valid,table,0,base,'nominal',None,.005)==4
    assert choose(None,grid,valid,table,0,base,'tolerant',None,.005) is None
    base={'success':False,'margin':-.02}
    # A known failed center must invalidate every tolerant window that contains it.
    index=choose(None,grid,valid,table,0,base,'tolerant',.03,.031)
    assert index is None


def test_tolerance_audit_keeps_goal_validity_and_separate_simulation_counts():
    from counterfactual.tolerance_experiment import run_parent
    from counterfactual.architecture import AnalyticPredictor
    config={'selectors':['nominal','tolerant'],'calibration_gamma_m':[.03],'audit_offsets_m':[-.005,-.0025,0.,.0025,.005],'epsilon_m':.005}
    record=run_parent(45001,{'analytic':AnalyticPredictor()},config,'calibration')
    selected=[r for r in record['rows'] if r['selected']]
    assert selected and record['calls']['center']>0 and record['calls']['offline_perturbation']>0
    assert all(len(r['checks'])==5 and 'valid' in r['checks'][2] for r in selected)
    assert all(r['logical_center_calls']==1 and r['logical_offline_offsets']==4 for r in selected)
