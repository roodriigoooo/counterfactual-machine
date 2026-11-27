"""Post-selection calibration on disjoint parents; independent audit thereafter."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .models import load_predictor
from .scenes import generate_scene,generate_goal
from .planning import plan
from .goals import goal_result
from .outcomes import BaselineOutcome
from .simulator import simulate
from .evaluation import interval

THRESHOLDS=[0.,.01,.03,.06,.1,.2,.4]


def choose_threshold(rows):
    eligible=[t for t in THRESHOLDS if (s:=summarize(rows[str(t)]))['proposals']>=10 and s['false_success']<=.1]
    return min(eligible,key=lambda t:(-summarize(rows[str(t)])['coverage'],t)) if eligible else None


def summarize(rows):
    proposals=sum(r['selected'] for r in rows);successes=sum(r['success'] for r in rows)
    return dict(tasks=len(rows),proposals=proposals,successes=successes,failures=proposals-successes,
                coverage=proposals/len(rows) if rows else 0,
                false_success=(proposals-successes)/proposals if proposals else None,
                false_success_ci=interval(rows,lambda r:r['selected'] and not r['success'],lambda r:r['selected']),
                by_kind={k:dict(tasks=sum(r['kind']==k for r in rows),proposals=sum(r['selected'] and r['kind']==k for r in rows),failures=sum(r['selected'] and not r['success'] and r['kind']==k for r in rows)) for k in ['reach','avoid']})


def run(checkpoint,output,parents=32):
    if (output/'calibration.json').exists():raise ValueError('preserve existing calibration; use another output path')
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);predictor=load_predictor(checkpoint)
    protocol=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),model=predictor.identity,
                  selection='cell minimizes mean validation vector RMSE across seeds; fixed seed 0, not best seed',
                  fit_parents=list(range(20000,20000+parents)),audit_parents=list(range(21000,21000+parents)),
                  goals_per_parent=8,thresholds=THRESHOLDS,rule='maximum selection coverage with empirical false-success <=0.10 and >=10 proposals; tie lowest threshold; no qualifying threshold means abstention',
                  goal_generator='generate_goal(scene_seed * 8 + index)',scope='calibration and independent calibration audit; not final research evaluation')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    all_rows={};tasks=[];calls={'original':0,'fit_verifications':0,'audit_verifications':0};selected_threshold=None
    for phase in ['fit','audit']:
        thresholds=THRESHOLDS if phase=='fit' else [selected_threshold]
        phase_rows={str(t):[] for t in thresholds}
        for seed in protocol[phase+'_parents']:
            scene=generate_scene(seed);original=simulate(scene);calls['original']+=1
            verified={}
            for j in range(8):
                goal=generate_goal(seed*8+j,scene.ball.id)
                tasks.append(dict(phase=phase,scene=scene.model_dump(),goal=goal.model_dump(),goal_index=j))
                base_result=goal_result(original.physics_times,original.physics_positions,goal)
                if base_result['success']:continue
                baseline=BaselineOutcome.from_verification(scene,goal,base_result)
                for threshold in thresholds:
                    result=plan(scene,goal,predictor,safety_margin=threshold,baseline=baseline) if threshold is not None else None
                    idx=result['selected_index'] if result else None
                    actual=None;u=None
                    if idx is not None:
                        u=float(result['grid'][idx])
                        if u not in verified:
                            verified[u]=simulate(scene,u);calls[phase+'_verifications']+=1
                        tr=verified[u];actual=goal_result(tr.physics_times,tr.physics_positions,goal)
                    phase_rows[str(threshold)].append(dict(parent_id=scene.parent_id,goal_index=j,kind=goal.kind,family=scene.family,selected=idx is not None,success=bool(actual and actual['success']),edit_m=u,logical_verification_calls=int(idx is not None)))
            print(phase,seed,flush=True)
        all_rows[phase]=phase_rows
        if phase=='fit':
            selected_threshold=choose_threshold(phase_rows)
            (output/'selection.json').write_text(json.dumps(dict(threshold_m=selected_threshold,fit={k:summarize(v) for k,v in phase_rows.items()}),indent=2))
    report=dict(protocol=protocol,selected_threshold_m=selected_threshold,fit={k:summarize(v) for k,v in all_rows['fit'].items()},audit=summarize(all_rows['audit'][str(selected_threshold)]),calls=calls,rows=all_rows,
                limits='Empirical calibration is not a safety guarantee; parent-bootstrap intervals condition on the selected checkpoint and threshold. No oracle feasibility filter; all initially failing tasks included.')
    (output/'calibration.json').write_text(json.dumps(report,indent=2));(output/'tasks.json').write_text(json.dumps(tasks,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','protocol','fit']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=Path('artifacts/mechanics/swept-local-release/seed-0/model.pt'));p.add_argument('--output',type=Path,default=Path('artifacts/mechanics/calibration'));p.add_argument('--parents',type=int,default=32);a=p.parse_args();run(a.checkpoint,a.output,a.parents)
