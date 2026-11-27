"""Fresh calibrated selector study; resumable at each independent parent."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .architecture import ArchitecturePredictor,AnalyticPredictor
from .architecture_evaluation import forecast_table
from .architecture_data import digest
from .tolerance import select_with_tolerance,successful_components
from .geometry import candidates,apply_edit
from .schemas import EditSpec
from .scenes import generate_scene,generate_goal
from .simulator import simulate
from .goals import goal_result


def choose(scene,grid,valid,table,j,base,selector,gamma,epsilon):
    zero=int(np.argmin(abs(grid)))
    if selector=='nominal' and base['success']:return zero
    if gamma is None:return None
    margins=table['margins'][j].copy();passed=table['passed'][j].copy()
    # This is the known original GOAL outcome, not an observed future supplied to dynamics.
    margins[zero]=base['margin'];passed[zero]=base['success']
    return select_with_tolerance(grid,margins,passed,valid,epsilon if selector=='tolerant' else 0.,gamma)['index']


def boundary_explanation(grid,component,trajectories):
    explanations=[]
    for side,inside,outside in [('left',component['first'],component['first']-1),('right',component['last'],component['last']+1)]:
        if outside<0 or outside>=len(grid):
            explanations.append(dict(side=side,kind='edit_bound',text='The permitted edit bound ends this sampled range.'));continue
        if trajectories[outside] is None:
            explanations.append(dict(side=side,kind='invalid_geometry',text='An invalid bumper position ends this sampled range.'));continue
        onset=lambda tr:[dict(object=next(o for o in e['objects'] if o!='ball'),time_s=e['time']) for e in tr.events if e['phase']=='begin']
        a,b=onset(trajectories[inside]),onset(trajectories[outside]);aa=[e['object'] for e in a];bb=[e['object'] for e in b]
        if aa!=bb:
            index=next((i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y),min(len(aa),len(bb)))
            before=aa[index] if index<len(aa) else 'no further contact';after=bb[index] if index<len(bb) else 'no further contact'
            text=f'Leaving the {side} edge changes contact {index+1}: {before} inside, {after} outside.';kind='contact_sequence'
        elif a:
            index=int(np.argmax([abs(x['time_s']-y['time_s']) for x,y in zip(a,b)]));shift=1000*(b[index]['time_s']-a[index]['time_s'])
            text=f'Contact order stays the same; the largest onset shift is {a[index]["object"]} ({shift:+.2f} ms). The goal boundary limits success; this timing association does not prove a unique cause.';kind='timing_association'
        else:text='No contact event changes here; the goal boundary limits sampled success.';kind='goal_boundary'
        explanations.append(dict(side=side,kind=kind,text=text,inside_edit_m=float(grid[inside]),outside_edit_m=float(grid[outside]),inside_events=a,outside_events=b))
    return explanations


def run_parent(seed,models,config,stage,thresholds=None):
    scene=generate_scene(seed);goals=[generate_goal(seed*8+j) for j in range(8)];grid,valid,_=candidates(scene,257)
    tables={name:forecast_table(model,scene,grid,valid,goals) for name,model in models.items()}
    original=simulate(scene);baseline=[goal_result(original.physics_times,original.physics_positions,g) for g in goals]
    calls=dict(original=1,center=0,offline_perturbation=0,oracle=0);centers={};perturbed={};rows=[];oracle=None;actual_tables=None;gallery=[]
    if stage=='evaluation':
        oracle=[]
        for i,u in enumerate(grid):
            oracle.append(simulate(scene,float(u)) if valid[i] else None);calls['oracle']+=int(valid[i])
        actual_tables=[[goal_result(tr.physics_times,tr.physics_positions,g) if tr else dict(success=False,margin=None) for tr in oracle] for g in goals]
    def audit(index,j):
        u=float(grid[index])
        if index not in centers:centers[index]=simulate(scene,u);calls['center']+=1
        outcomes=[]
        for offset in config['audit_offsets_m']:
            shifted=u+offset
            try:apply_edit(scene,EditSpec(displacement=shifted))
            except ValueError:
                outcomes.append(dict(offset_m=offset,valid=False,success=False,margin=None));continue
            if offset==0:tr=centers[index]
            else:
                key=round(shifted,12)
                if key not in perturbed:perturbed[key]=simulate(scene,shifted);calls['offline_perturbation']+=1
                tr=perturbed[key]
            actual=goal_result(tr.physics_times,tr.physics_positions,goals[j]);outcomes.append(dict(offset_m=offset,**actual))
        return outcomes
    for name,table in tables.items():
        for selector in config['selectors']:
            key=f'{name}:{selector}'
            settings=[(str(g),g) for g in config['calibration_gamma_m']] if stage=='calibration' else [('fixed',config['fixed_gamma_m']),('calibrated',thresholds[key])]
            for mode,gamma in settings:
                for j,goal in enumerate(goals):
                    base=baseline[j]
                    if stage=='calibration' and base['success']:continue
                    index=choose(scene,grid,valid,table,j,base,selector,gamma,config['epsilon_m'])
                    checks=audit(index,j) if index is not None else []
                    center=checks[2] if checks else None
                    row=dict(parent_id=scene.parent_id,goal_index=j,kind=goal.kind,model=name,selector=selector,mode=mode,initial_success=base['success'],selected=index is not None,
                             edit_m=float(grid[index]) if index is not None else None,edit_radii=abs(float(grid[index]))/scene.ball.radius if index is not None else None,
                             center_success=bool(center and center['success']),perturbed_success=bool(checks and all(r['success'] for r in checks)),
                             perturbation_success_fraction=float(np.mean([r['success'] for r in checks])) if checks else 0.,
                             worst_margin_m=min(r['margin'] for r in checks) if checks and all(r['valid'] for r in checks) else None,
                             checks=checks,logical_center_calls=int(index is not None),logical_offline_offsets=4*int(index is not None))
                    if stage=='evaluation':
                        actual=actual_tables[j];passed=np.array([r['success'] for r in actual]);margins=np.array([r['margin'] if r['margin'] is not None else np.nan for r in actual])
                        nominal_oracle=select_with_tolerance(grid,margins,passed,valid,0,0)['index'];robust_oracle=select_with_tolerance(grid,margins,passed,valid,config['epsilon_m'],0)['index']
                        row.update(grid_solvable=nominal_oracle is not None,robust_grid_solvable=robust_oracle is not None,
                                   nominal_oracle_cost=abs(float(grid[nominal_oracle]))/scene.ball.radius if nominal_oracle is not None else None,
                                   tolerant_oracle_cost=abs(float(grid[robust_oracle]))/scene.ball.radius if robust_oracle is not None else None)
                        if index is not None:
                            components=successful_components(grid,passed);actual_component=next((x for x in components if x['first']<=index<=x['last']),None)
                            predicted_components=successful_components(grid,table['passed'][j]&valid)
                            predicted_component=next((x for x in predicted_components if x['first']<=index<=x['last']),None)
                            row['boundary_error_mm']=max(abs(predicted_component[k]-actual_component[k])*1000 for k in ['low_m','high_m']) if actual_component and predicted_component else None
                            if name=='analytic' and selector=='tolerant' and mode=='fixed' and not base['success']:
                                nominal_idx=choose(scene,grid,valid,table,j,base,'nominal',gamma,config['epsilon_m']);nominal_checks=audit(nominal_idx,j) if nominal_idx is not None else []
                                interesting=row['perturbed_success'] and nominal_checks and not all(r['success'] for r in nominal_checks)
                                failure=not row['perturbed_success']
                                if interesting or failure:
                                    explanation=boundary_explanation(grid,actual_component,oracle) if actual_component else []
                                    has_sequence=any(e['kind']=='contact_sequence' for e in explanation)
                                    # At most one per category per parent; global first-order selection follows later.
                                    category='sequence' if interesting and has_sequence else 'success' if interesting else 'failure'
                                    if not any(g['category']==category for g in gallery):
                                        position=int(np.flatnonzero(np.flatnonzero(valid)==index)[0])
                                        gallery.append(dict(category=category,scene=scene.model_dump(),goal=goal.model_dump(),row=row,grid_m=grid.tolist(),valid=valid.tolist(),actual_margins=[r['margin'] for r in actual],predicted_margins=[float(v) if np.isfinite(v) else None for v in table['margins'][j]],
                                            components=components,measured_component=actual_component,boundaries=explanation,nominal_edit_m=float(grid[nominal_idx]) if nominal_idx is not None else None,
                                            nominal_checks=nominal_checks,original=original.public(),verified=centers[index].public(),predicted=table['states'][position].tolist(),epsilon_m=config['epsilon_m'],effective_grid_epsilon_m=float(np.ceil(config['epsilon_m']/(grid[1]-grid[0])-1e-10)*(grid[1]-grid[0]))))
                    rows.append(row)
    return dict(scene_seed=seed,stage=stage,rows=rows,calls=calls,gallery=gallery,latencies={k:dict(cold_s=v['cold_s'],table_s=v['table_s']) for k,v in tables.items()})


def experiment(root,protocol):
    torch.set_num_threads(4);root.mkdir(parents=True,exist_ok=True);config=json.loads(protocol.read_text());models={};hashes={}
    for seed in config['seeds']:
        cp=Path(f'artifacts/contact-learning/{config["learned_cell"]}/seed-{seed}/model.pt');name=f'learned-{seed}';models[name]=ArchitecturePredictor(cp);hashes[name]=digest(cp)
    models['analytic']=AnalyticPredictor();binding=dict(protocol_sha256=digest(protocol),checkpoints=hashes)
    frozen=root/'frozen-inputs.json'
    if frozen.exists() and json.loads(frozen.read_text())!=binding:raise ValueError('frozen selector inputs changed')
    if not frozen.exists():frozen.write_text(json.dumps(binding,indent=2));(root/'protocol.json').write_bytes(protocol.read_bytes())
    thresholds=None
    for stage in ['calibration','evaluation']:
        folder=root/f'{stage}-parents';folder.mkdir(exist_ok=True);start,count=config['parent_pools'][stage];records=[]
        for seed in range(start,start+count):
            path=folder/f'{seed}.json'
            if path.exists():record=json.loads(path.read_text())
            else:
                record=run_parent(seed,models,config,stage,thresholds);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record));temporary.replace(path);print(stage,seed,flush=True)
            records.append(record)
        if stage=='calibration':
            thresholds={};summaries={};rows=[r for p in records for r in p['rows']]
            for name in models:
                for selector in config['selectors']:
                    key=f'{name}:{selector}';options=[];summaries[key]={}
                    for gamma in config['calibration_gamma_m']:
                        group=[r for r in rows if r['model']==name and r['selector']==selector and r['mode']==str(gamma)];accepted=[r for r in group if r['selected']];fail=sum(not r['perturbed_success'] for r in accepted)
                        summaries[key][str(gamma)]=dict(tasks=len(group),proposals=len(accepted),perturbed_failures=fail,false_success=fail/len(accepted) if accepted else None)
                        if len(accepted)>=10 and fail/len(accepted)<=.1:options.append((len(accepted),gamma))
                    thresholds[key]=min(options,key=lambda x:(-x[0],x[1]))[1] if options else None
            (root/'calibration.json').write_text(json.dumps(dict(thresholds=thresholds,summary=summaries),indent=2))
        else:
            examples=[g for p in records for g in p['gallery']];chosen=next((g for g in examples if g['category']=='sequence'),next((g for g in examples if g['category']=='success'),None));failure=next((g for g in examples if g['category']=='failure'),None)
            (root/'gallery.json').write_text(json.dumps(dict(success=chosen,failure=failure),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/tolerance-study'));p.add_argument('--protocol',type=Path,default=Path('configs/contact-tolerance.json'));a=p.parse_args();experiment(a.root,a.protocol)
