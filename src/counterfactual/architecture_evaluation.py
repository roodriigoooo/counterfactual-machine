"""Frozen-checkpoint calibration and fresh final factorial evaluation.

Predicted tables are computed before oracle/verification outcomes. The planner
uses only predicted margins and the shared known-original guard. Oracle paths
never enter predictors. Actual simulator work is shared; logical per-method
verification budgets are retained separately.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .architecture import ArchitecturePredictor, AnalyticPredictor
from .architecture_data import digest
from .geometry import candidates
from .geometry import TOLERANCE
from .scenes import generate_scene, generate_goal
from .schemas import SceneSpec
from .goals import goal_result
from .planning import select_index
from .simulator import simulate
from .reference import sweep
from .evaluation import interval
from .mechanics_evaluation import measure, suite_measure


def selection(grid, margins, passed, threshold):
    if threshold is None:return None
    success=passed & (margins>=threshold)
    success[int(np.argmin(abs(grid)))]=False  # all evaluated tasks have verified failed originals
    return select_index(grid,margins,success)


def forecast_table(predictor,scene,grid,valid,goals):
    started=perf_counter();states,scores=predictor.forecast(scene,grid[valid]);cold=perf_counter()-started
    margins=np.full((len(goals),len(grid)),np.nan);passed=np.zeros_like(margins,dtype=bool)
    times=np.linspace(0,2,121);indices=np.flatnonzero(valid)
    numerical=np.isfinite(states).all(axis=(1,2)) & (abs(states[:,:,:2]).max(axis=(1,2))<100)
    for k,goal in enumerate(goals):
        if goal.region.kind=='disk' and goal.start==0 and goal.end==2:
            # Exact same polyline distance as goal_result, vectorized over candidates.
            a=states[:,:-1,:2].astype(np.float64)-goal.region.center
            b=states[:,1:,:2].astype(np.float64)-goal.region.center
            d=b-a;u=np.clip(-np.sum(a*d,axis=-1)/np.maximum(np.sum(d*d,axis=-1),1e-30),0,1)
            distance=np.linalg.norm(a+u[:,:,None]*d,axis=-1).min(1)-goal.region.radius
            m=-distance if goal.kind=='reach' else distance
            ok=m>=-TOLERANCE if goal.kind=='reach' else m>TOLERANCE
            margins[k,indices[numerical]]=m[numerical];passed[k,indices[numerical]]=ok[numerical]
        else:
            for j,i in enumerate(indices):
                if not numerical[j]:continue
                result=goal_result(times,states[j,:,:2],goal)
                margins[k,i]=result['margin'];passed[k,i]=result['success']
    return dict(margins=margins,passed=passed,states=states,scores=scores,cold_s=cold,table_s=perf_counter()-started)


def summarize(rows):
    n=len(rows);selected=sum(r['selected'] for r in rows);success=sum(r['success'] for r in rows)
    result=dict(tasks=n,proposals=selected,successes=success,failures=selected-success,
                coverage=selected/max(1,n),false_success=(selected-success)/selected if selected else None)
    if rows and 'solvable' in rows[0]:
        solvable=[r for r in rows if r['solvable']]
        result.update(solvable_tasks=len(solvable),success_solvable=interval(solvable,lambda r:r['success'],lambda r:1),
                      coverage_ci=interval(rows,lambda r:r['selected'],lambda r:1),
                      false_success_ci=interval(rows,lambda r:r['selected'] and not r['success'],lambda r:r['selected']))
        excess=[r['edit_cost']-r['oracle_cost'] for r in rows if r['success'] and r['oracle_cost'] is not None]
        result['edit_excess_radii']=dict(n=len(excess),mean=float(np.mean(excess)) if excess else None,p95=float(np.quantile(excess,.95)) if excess else None)
    return result


def fit(models,config,root):
    thresholds=config['thresholds_m'];rows={name:{str(t):[] for t in thresholds} for name in models}
    start,n=config['parent_pools']['calibration'];calls=0;original_calls=0
    for seed in range(start,start+n):
        scene=generate_scene(seed);grid,valid,_=candidates(scene,config['grid_candidates'])
        goals=[generate_goal(seed*8+j) for j in range(8)];original=simulate(scene);original_calls+=1
        failed=[j for j,g in enumerate(goals) if not goal_result(original.physics_times,original.physics_positions,g)['success']]
        tables={name:forecast_table(p,scene,grid,valid,goals) for name,p in models.items()}
        verified={}
        for name,table in tables.items():
            for t in thresholds:
                for j in failed:
                    idx=selection(grid,table['margins'][j],table['passed'][j],t);actual=None
                    if idx is not None:
                        if idx not in verified:verified[idx]=simulate(scene,float(grid[idx]));calls+=1
                        tr=verified[idx];actual=goal_result(tr.physics_times,tr.physics_positions,goals[j])
                    rows[name][str(t)].append(dict(parent_id=scene.parent_id,goal_index=j,kind=goals[j].kind,family=scene.family,selected=idx is not None,
                                                 success=bool(actual and actual['success']),edit_m=float(grid[idx]) if idx is not None else None))
        print('calibration',seed,flush=True)
    selected={};summaries={}
    for name,by_t in rows.items():
        summaries[name]={t:summarize(r) for t,r in by_t.items()}
        eligible=[float(t) for t,s in summaries[name].items() if s['proposals']>=10 and s['false_success']<=.1]
        selected[name]=min(eligible,key=lambda t:(-summaries[name][str(t)]['coverage'],t)) if eligible else None
    record=dict(selected_thresholds_m=selected,summary=summaries,rows=rows,actual_original_calls=original_calls,actual_shared_verification_calls=calls)
    (root/'calibration.json').write_text(json.dumps(record,indent=2))
    return record


def final(models,config,calibration,root):
    rows={name:{'fixed':[],'calibrated':[]} for name in models};latencies={name:[] for name in models}
    gallery={};calls=dict(original=0,oracle=0,shared_verification=0);tasks=[]
    start,n=config['parent_pools']['test']
    for seed in range(start,start+n):
        scene=generate_scene(seed);grid,valid,_=candidates(scene,config['grid_candidates'])
        goals=[generate_goal(seed*8+j) for j in range(8)]
        tables={name:forecast_table(p,scene,grid,valid,goals) for name,p in models.items()}
        for name,table in tables.items():latencies[name].append(dict(parent_id=scene.parent_id,cold_forecast_s=table['cold_s'],eight_goal_table_s=table['table_s']))
        oracle=sweep(scene,goals[0],count=config['grid_candidates']);calls['oracle']+=oracle['simulator_calls']
        original=simulate(scene);calls['original']+=1;verified={}
        if 'contact_transition' not in gallery and oracle['bumper_contacts'][valid].any() and not oracle['bumper_contacts'][valid].all():
            times=np.linspace(0,2,121);bi=0
            gallery['contact_transition']=dict(scene=scene.model_dump(),grid=grid.tolist(),valid=valid.tolist(),contact=oracle['bumper_contacts'].tolist(),
                final_dx=[float(tr.states[-1,0]-original.states[-1,0]) if tr else None for tr in oracle['trajectories']],
                predicted_final_dx={name:(table['states'][:,-1,0]-table['states'][np.flatnonzero(grid[valid]==0)[0],-1,0]).tolist() for name,table in tables.items() if name.endswith('0') or name=='analytic'})
        for j,goal in enumerate(goals):
            base=goal_result(original.physics_times,original.physics_positions,goal)
            tasks.append(dict(scene=scene.model_dump(),goal=goal.model_dump(),goal_index=j,original_success=base['success']))
            if base['success']:continue
            actual=[goal_result(t.physics_times,t.physics_positions,goal) if t else dict(success=False,margin=None) for t in oracle['trajectories']]
            feasible=np.array([r['success'] for r in actual]);passing=np.flatnonzero(feasible)
            cost=float(abs(grid[passing]).min()/scene.ball.radius) if len(passing) else None
            for name,table in tables.items():
                for mode,threshold in [('fixed',config['fixed_margin_m']),('calibrated',calibration['selected_thresholds_m'][name])]:
                    idx=selection(grid,table['margins'][j],table['passed'][j],threshold);result=None
                    if idx is not None:
                        if idx not in verified:verified[idx]=simulate(scene,float(grid[idx]));calls['shared_verification']+=1
                        tr=verified[idx];result=goal_result(tr.physics_times,tr.physics_positions,goal)
                    row=dict(parent_id=scene.parent_id,goal_index=j,kind=goal.kind,family=scene.family,solvable=bool(len(passing)),selected=idx is not None,
                             success=bool(result and result['success']),edit_cost=float(abs(grid[idx])/scene.ball.radius) if idx is not None else None,
                             edit_m=float(grid[idx]) if idx is not None else None,oracle_cost=cost,predicted_margin_m=float(table['margins'][j,idx]) if idx is not None else None,
                             verified_margin_m=result['margin'] if result else None,logical_verification_calls=int(idx is not None),expected_random_success=float(feasible[valid].mean()))
                    rows[name][mode].append(row)
                    if name=='A/seed-0' and mode=='fixed' and idx is not None:
                        category='success' if row['success'] else 'failure'
                        if category not in gallery:
                            pos=int(np.flatnonzero(np.flatnonzero(valid)==idx)[0])
                            gallery[category]=dict(scene=scene.model_dump(),goal=goal.model_dump(),edit_m=float(grid[idx]),row=row,grid=grid.tolist(),
                              actual_margins=[r['margin'] for r in actual],predicted_margins={k:[float(v) if np.isfinite(v) else None for v in t['margins'][j]] for k,t in tables.items() if k.endswith('0') or k=='analytic'},
                              original=original.states.tolist(),verified=verified[idx].states.tolist(),predicted=table['states'][pos].tolist())
        print('final',seed,flush=True)
    result=dict(models={name:{mode:dict(summary=summarize(r),rows=r) for mode,r in modes.items()} for name,modes in rows.items()},calls=calls,latencies=latencies)
    (root/'decisions.json').write_text(json.dumps(result,indent=2));(root/'gallery.json').write_text(json.dumps(gallery,indent=2));(root/'tasks.json').write_text(json.dumps(tasks,indent=2))
    return result


def evaluate(root,data,protocol):
    torch.set_num_threads(4);config=json.loads(protocol.read_text());models={};hashes={}
    for variant in config['variants']:
        for seed in config['seeds']:
            p=root/variant/f'seed-{seed}'/'model.pt';name=f'{variant}/seed-{seed}'
            if not (p.parent/'training.json').exists():raise ValueError(f'incomplete training: {name}')
            payload=torch.load(p,map_location='cpu',weights_only=True)
            if payload['protocol_sha256']!=digest(protocol):raise ValueError('checkpoint/protocol mismatch')
            models[name]=ArchitecturePredictor(p);hashes[name]=digest(p)
    models['analytic']=AnalyticPredictor()
    frozen=root/'frozen-checkpoints.json'
    record=dict(protocol_sha256=digest(protocol),checkpoints=hashes)
    if frozen.exists() and json.loads(frozen.read_text())!=record:raise ValueError('evaluation inputs changed')
    frozen.write_text(json.dumps(record,indent=2))
    fitpath=root/'calibration.json';calibration=json.loads(fitpath.read_text()) if fitpath.exists() else fit(models,config,root)
    if not (root/'decisions.json').exists():final(models,config,calibration,root)
    for name,predictor in models.items():
        out=root/'physical'/f"{name.replace('/','-')}.json"
        if out.exists():continue
        out.parent.mkdir(exist_ok=True)
        physical=dict(validation=measure(predictor,data/'validation','validation'),test=measure(predictor,data/'test','test'),
                      suite=suite_measure(predictor,Path('artifacts/contact-suite/suite.json')))
        out.write_text(json.dumps(physical,indent=2));print('physical',name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/architecture'));p.add_argument('--data',type=Path,default=Path('data/architecture'));p.add_argument('--protocol',type=Path,default=Path('configs/architecture-experiment.json'));a=p.parse_args();evaluate(a.root,a.data,a.protocol)
