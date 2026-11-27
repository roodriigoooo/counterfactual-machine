"""Common finite-grid development evaluation; cache simulator paths across models."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .models import load_predictor
from .schemas import SceneSpec,GoalSpec
from .planning import plan
from .outcomes import BaselineOutcome
from .goals import goal_result
from .reference import sweep
from .simulator import simulate
from .evaluation import interval


def evaluate(root):
    torch.set_num_threads(4)
    models={str(p.parent.relative_to(root)):load_predictor(p) for p in sorted(root.glob('*/seed-*/model.pt')) if (p.parent/'training.json').exists()}
    tasks=json.loads(Path('artifacts/pilot/tasks.json').read_text())
    rows={name:[] for name in models};oracle_calls=0
    for parent in sorted({t['scene']['parent_id'] for t in tasks}):
        local=[t for t in tasks if t['scene']['parent_id']==parent];s=SceneSpec.model_validate(local[0]['scene'])
        oracle=sweep(s,GoalSpec.model_validate(local[0]['goal']));oracle_calls+=oracle['simulator_calls']
        zero=oracle['trajectories'][128]
        for task in local:
            goal=GoalSpec.model_validate(task['goal']);baseline_result=goal_result(zero.physics_times,zero.physics_positions,goal)
            if baseline_result['success']:continue
            outcomes=[goal_result(tr.physics_times,tr.physics_positions,goal) if tr else dict(success=False,margin=None) for tr in oracle['trajectories']]
            success=np.array([v['success'] for v in outcomes]);passing=np.flatnonzero(success)
            cost=min(abs(oracle['grid'][passing]))/.12 if len(passing) else None
            baseline=BaselineOutcome.from_verification(s,goal,baseline_result)
            for name,p in models.items():
                result=plan(s,goal,p,baseline=baseline);idx=result['selected_index'];actual=None
                if idx is not None:
                    verified=simulate(s,float(result['grid'][idx]));actual=goal_result(verified.physics_times,verified.physics_positions,goal)
                rows[name].append(dict(parent_id=parent,goal_kind=goal.kind,goal_index=task['goal_index'],family=s.family,solvable=bool(len(passing)),selected=idx is not None,success=bool(actual and actual['success']),edit_cost=abs(float(result['grid'][idx]))/.12 if idx is not None else None,oracle_cost=float(cost) if cost is not None else None,expected_random_success=float(success[oracle['valid']].mean()),latency_s=result['seconds'],verification_calls=int(idx is not None)))
        print('decisions',parent,flush=True)
    report={}
    for name,r in rows.items():
        solvable=[v for v in r if v['solvable']]
        summary=dict(successes=sum(v['success'] for v in r),failed_proposals=sum(v['selected'] and not v['success'] for v in r),proposals=sum(v['selected'] for v in r),failing_tasks=len(r),solvable_tasks=len(solvable),verified_success_solvable=interval(solvable,lambda v:v['success'],lambda v:1),false_success=interval(r,lambda v:v['selected'] and not v['success'],lambda v:v['selected']),coverage=interval(r,lambda v:v['selected'],lambda v:1))
        report[name]=dict(summary=summary,tasks=r)
    (root/'decisions.json').write_text(json.dumps(dict(cohort='existing pilot development, no final-test claim',threshold_m=.03,grid=257,oracle_rollouts_shared=oracle_calls,models=report),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/mechanics'));a=p.parse_args();evaluate(a.root)
