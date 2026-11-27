"""Post-evaluation numerical audit of the fixed-order illustration, never selection."""
import argparse,json
from pathlib import Path
import numpy as np
from .schemas import SceneSpec,GoalSpec
from .simulator import simulate
from .goals import goal_result
from .tolerance import successful_components
from .architecture_data import digest


def audit(root):
    gallery=json.loads((root/'gallery.json').read_text());records={}
    for category,g in gallery.items():
        if not g:continue
        scene=SceneSpec.model_validate(g['scene']);goal=GoalSpec.model_validate(g['goal']);grid=np.array(g['grid_m']);fine=[];trajectories={}
        for i,u in enumerate(grid):
            if not g['valid'][i]:fine.append(dict(valid=False,success=False,margin=None));continue
            tr=simulate(scene,float(u),physics_hz=3840);fine.append(goal_result(tr.physics_times,tr.physics_positions,goal));trajectories[i]=tr
        common=np.array([r['success'] and m is not None and (m>=0 if goal.kind=='reach' else m>0) for r,m in zip(fine,g['actual_margins'])]);components=successful_components(grid,common)
        index=int(np.argmin(abs(grid-g['row']['edit_m'])));component=next((c for c in components if c['first']<=index<=c['last']),None)
        checks=[]
        for offset in [-.005,-.0025,0.,.0025,.005]:
            tr=simulate(scene,g['row']['edit_m']+offset,physics_hz=3840)
            checks.append(dict(offset_m=offset,**goal_result(tr.physics_times,tr.physics_positions,goal)))
        boundaries=[]
        if component:
            for side,inside,outside in [('left',component['first'],component['first']-1),('right',component['last'],component['last']+1)]:
                if outside not in trajectories:
                    boundaries.append(dict(side=side,kind='edit_bound' if outside<0 or outside>=len(grid) else 'invalid_geometry'));continue
                records_at_rate={}
                for hz in [1920,3840]:
                    events={}
                    for label,i in [('inside',inside),('outside',outside)]:
                        tr=trajectories[i] if hz==3840 else simulate(scene,float(grid[i]),physics_hz=hz)
                        events[label]=[dict(time_s=e['time'],object=next(o for o in e['objects'] if o!=scene.ball.id),phase=e['phase']) for e in tr.events]
                    records_at_rate[str(hz)]=events
                boundaries.append(dict(side=side,kind='contact_association',inside_edit_m=float(grid[inside]),outside_edit_m=float(grid[outside]),events=records_at_rate))
        coarse=np.array([m is not None and (m>=0 if goal.kind=='reach' else m>0) for m in g['actual_margins']]);fine_pass=np.array([r['success'] for r in fine])
        records[category]=dict(scene=g['scene'],goal=g['goal'],edit_m=g['row']['edit_m'],epsilon_m=.005,grid_m=g['grid_m'],fine_margins=[r['margin'] for r in fine],common_success=common.tolist(),common_component=component,
            fine_checks=checks,coarse_checks=g['row']['checks'],goal_disagreement_count=int(((coarse!=fine_pass)&np.array(g['valid'])).sum()),boundaries=boundaries,
            explanation='The left boundary is associated with a change in the repeated bumper-contact episode. Callback counts and timing change with timestep; this is not evidence that one particular callback causes failure. The right boundary is the allowed edit limit.' if category=='success' else 'The predicted tolerance window fails verification, including its center. No measured successful range contains the proposed move.',
            qualifier='Successful sampled positions at both 1920 and 3840 Hz; grid spacing 0.9375 mm. Five execution offsets are checked separately. This is not a continuous tolerance certificate or a numerically invariant event sequence.')
        print(category,component,records[category]['goal_disagreement_count'],flush=True)
    result=dict(gallery_sha256=digest(root/'gallery.json'),role='Post-evaluation illustration audit; no selection or model changes',physics_hz=[1920,3840],examples=records)
    (root/'demonstration.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/tolerance-study'));a=p.parse_args();audit(a.root)
