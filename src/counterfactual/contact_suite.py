"""T1 fixed contact fixtures and tolerance-aware event matching."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from .schemas import SceneSpec,Ball,Bumper,Segment
from .simulator import simulate
from .recurrent import pack
from .contact_mechanics import swept_admission


def fixtures():
    floor=Segment(id='floor',a=(.1,.18),b=(5.9,.18))
    specs=[('clear_miss',(1.,3.),(0.,0.),(4.,2.),()),
           ('grazing_miss',(3.441,3.),(0.,0.),(3.,2.),()),
           ('separating',(3.45,2.),(3.,0.),(3.,2.),()),
           ('head_on',(3.,3.),(0.,0.),(3.,2.),()),
           ('oblique',(2.6,3.),(1.1,0.),(3.,2.),()),
           ('ramp',(1.5,3.),(1.,0.),(4.8,1.5),(floor,Segment(id='ramp',a=(1.,2.5),b=(3.,1.7)))),
           ('floor_bounce',(1.,2.),(1.,0.),(4.8,2.),(floor,)),
           ('repeated_contact',(1.,1.),(0.,0.),(4.8,2.),(floor,)),
           ('irrelevant_edit',(1.,3.),(.2,0.),(5.,2.),(floor,))]
    return [SceneSpec(id=name,parent_id='fixture-'+name,family='authored',ball=Ball(position=p,velocity=v),bumper=Bumper(position=b),segments=segments) for name,p,v,b,segments in specs]


def match_events(truth,predicted,tolerance=1/30):
    used_t=set();used_p=set();matched=[]
    for error,i,j in sorted((abs(t-p),i,j) for i,t in enumerate(truth) for j,p in enumerate(predicted)):
        if error<=tolerance and i not in used_t and j not in used_p:
            used_t.add(i);used_p.add(j);matched.append(error)
    return dict(matched=len(matched),invented=len(predicted)-len(used_p),missed=len(truth)-len(used_t),absolute_timing_errors_s=matched,tolerance_s=tolerance)


def build(output):
    output.mkdir(parents=True,exist_ok=True);records=[]
    for scene in fixtures():
        for edit in [-.12,0.,.12]:
            # A separating fixture may have an invalid overlapping positive edit.
            try:a=simulate(scene,edit,record_responses=True)
            except ValueError as e:
                records.append(dict(scene=scene.model_dump(),edit_m=edit,valid=False,reason=str(e)));continue
            b=simulate(scene,edit,3840,record_responses=True)
            from .geometry import apply_edit
            from .schemas import EditSpec
            _,context=pack(apply_edit(scene,EditSpec(displacement=edit)))
            with torch.no_grad():
                eligible,_,_,_=swept_admission(torch.tensor(a.states[:-1],dtype=torch.float32),torch.tensor(context[None]).expand(120,-1,-1))
            onset=lambda tr:[e for e in tr.events if e['phase']=='begin']
            delta=float(np.linalg.norm(a.states[:,:2]-b.states[:,:2],axis=1).max())
            records.append(dict(scene=scene.model_dump(),scene_hash=scene.content_hash(),edit_m=edit,valid=True,states=a.states.tolist(),contacts=a.contacts.tolist(),object_ids=a.object_ids,onsets=onset(a),responses=a.responses,configuration=a.provenance,comparison_onsets=onset(b),step_halving_max_position_m=delta,numerically_sensitive=delta>.05 or len(onset(a))!=len(onset(b)),multiple_candidate_intervals=int((eligible.sum(-1)>1).sum())))
    payload=dict(version='contact-suite-v1',scope='diagnostic fixtures, not training or final evaluation parents',records=records)
    payload['content_sha256']=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    (output/'suite.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(dict(hash=payload['content_sha256'],valid=sum(r['valid'] for r in records),sensitive=sum(r.get('numerically_sensitive',False) for r in records)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/contact-suite'));a=p.parse_args();build(a.output)
