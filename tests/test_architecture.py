import numpy as np
import torch
from counterfactual.architecture import ArchitectureModel, AnalyticPredictor, conditions, integrate
from counterfactual.architecture_training import variance_covariance
from counterfactual.contact_suite import fixtures
from counterfactual.contact_mechanics import MechanicsModel
from counterfactual.recurrent import pack
from counterfactual.geometry import apply_edit
from counterfactual.schemas import EditSpec


def test_same_capacity_and_zero_edit_for_all_variants():
    s,c=pack(fixtures()[3]);s=torch.tensor(s[None]);c=torch.tensor(c[None]);counts=[]
    for variant in 'ABCD':
        torch.manual_seed(0);m=ArchitectureModel(variant);counts.append(sum(p.numel() for p in m.parameters()))
        a,b=m.paired(s,c,c,steps=12)
        torch.testing.assert_close(a['states'],b['states'],rtol=0,atol=0)
    assert len(set(counts))==1


def test_difference_is_context_dependent_and_receives_gradient():
    torch.manual_seed(0);m=ArchitectureModel('B');a=torch.randn(3,5,32);b=torch.randn(3,5,32)
    assert not torch.allclose(m.response(a,b),m.response(a,a))
    m.response(a,b).sum().backward()
    assert m.contextual[0].weight.grad[:,64:].abs().sum()>0


def test_prior_matches_preserved_mechanics_integrator():
    m=MechanicsModel('swept','local').eval();scene=fixtures()[3];s,c=pack(scene);s[:4]=[3.,2.45,0.,-3.]
    s=torch.tensor(s[None]);c=torch.tensor(c[None]);memory=torch.zeros(1,5,32)
    old,h,_=m.step(s,c,m.encode(c),memory)
    _,geo=conditions(s,c);new,_=integrate(s,c,geo,m.response(h)[...,1:3].sigmoid())
    torch.testing.assert_close(old,new,rtol=0,atol=0)


def test_analytic_control_has_real_response_and_no_invented_miss_effect():
    p=AnalyticPredictor();scene=fixtures()[3];x,_=p.forecast(scene,[0]);assert np.isfinite(x).all()
    scene=fixtures()[0];x,_=p.forecast(scene,[0,.02]);np.testing.assert_array_equal(x[0],x[1])


def test_anti_collapse_penalty_detects_constant_embedding():
    variance,cov=variance_covariance(torch.zeros(8,16))
    assert variance>.9 and cov==0


def test_cached_goal_selection_matches_public_planner():
    from counterfactual.architecture_evaluation import forecast_table,selection
    from counterfactual.geometry import candidates
    from counterfactual.planning import plan
    from counterfactual.outcomes import BaselineOutcome
    from counterfactual.scenes import generate_scene,generate_goal
    from counterfactual.simulator import simulate
    from counterfactual.goals import goal_result
    scene=generate_scene(32000);predictor=AnalyticPredictor();tr=simulate(scene)
    goals=[generate_goal(scene.seed*8+j) for j in range(8)]
    grid,valid,_=candidates(scene);table=forecast_table(predictor,scene,grid,valid,goals)
    checked=0
    for j,goal in enumerate(goals):
        actual=goal_result(tr.physics_times,tr.physics_positions,goal)
        if actual['success']:continue
        baseline=BaselineOutcome.from_verification(scene,goal,actual)
        for threshold in [0,.03,.1]:
            expected=plan(scene,goal,predictor,safety_margin=threshold,baseline=baseline)['selected_index']
            assert selection(grid,table['margins'][j],table['passed'][j],threshold)==expected
            checked+=1
    assert checked>0


def test_fresh_parent_pools_are_disjoint_and_data_labels_are_consistent():
    import json
    from pathlib import Path
    config=json.loads(Path('configs/architecture-experiment.json').read_text())
    pools=[set(range(start,start+count)) for start,count in config['parent_pools'].values()]
    assert sum(map(len,pools))==len(set.union(*pools))
    assert not set.union(*pools)&(set(range(256))|set(range(20000,20032))|set(range(21000,21032)))
    path=Path('data/architecture/train/paired.npz')
    if not path.exists():return
    with np.load(path) as a:
        active=(a['phase']==1)|(a['phase']==2)
        np.testing.assert_array_equal(active,a['contacts'].astype(bool))
        assert np.all((a['onset'][a['onset_mask']>0]>0)&(a['onset'][a['onset_mask']>0]<=1+1e-6))
        assert np.max(np.abs(a['response'][a['contacts']==0]))==0


def test_difference_branch_creates_new_collision_but_not_premature_motion():
    scene=fixtures()[1]
    edited=apply_edit(scene,EditSpec(displacement=.12))
    state,c0=pack(scene);_,cu=pack(edited)
    model=ArchitectureModel('B')
    a,b=model.paired(torch.tensor(state[None]),torch.tensor(c0[None]),torch.tensor(cu[None]))
    torch.testing.assert_close(a['states'][:,:15],b['states'][:,:15],rtol=0,atol=0)
    assert not torch.allclose(a['hidden'][:,0],b['hidden'][:,0])  # anticipatory context may change
    assert (a['states']-b['states']).abs().max()>1  # a newly created encounter has a real effect


def test_difference_predictor_reuses_exact_original_across_batch_sizes(tmp_path):
    from counterfactual.architecture import ArchitecturePredictor
    model=ArchitectureModel('B');path=tmp_path/'model.pt'
    torch.save(dict(model=model.state_dict(),variant='B',name='Structured architecture experiment',seed=0,epoch=0,dataset_sha256='fixture'),path)
    predictor=ArchitecturePredictor(path);scene=fixtures()[1]
    a,_=predictor.forecast(scene,[0]);b,_=predictor.forecast(scene,[0,.02,.12])
    np.testing.assert_array_equal(a[0],b[0]);assert len(predictor.originals)==1


def test_vectorized_goal_table_preserves_polyline_margins():
    from counterfactual.architecture_evaluation import forecast_table
    from counterfactual.geometry import candidates
    from counterfactual.scenes import generate_scene,generate_goal
    from counterfactual.goals import goal_result
    scene=generate_scene(32001);p=AnalyticPredictor();grid,valid,_=candidates(scene,9)
    goals=[generate_goal(scene.seed*8+j) for j in range(8)]
    table=forecast_table(p,scene,grid,valid,goals)
    for j,goal in enumerate(goals):
        for k,i in enumerate(np.flatnonzero(valid)):
            expected=goal_result(np.linspace(0,2,121),table['states'][k,:,:2],goal)
            assert abs(table['margins'][j,i]-expected['margin'])<1e-7
            assert table['passed'][j,i]==expected['success']
