"""New parents with ordinary, support, repeat, miss and competing-contact strata."""
import argparse
import json
from pathlib import Path
import numpy as np
from .schemas import Ball,Bumper,SceneSpec,Segment,Material,EditSpec
from .scenes import generate_scene
from .geometry import validate_scene,candidates,apply_edit
from .simulator import simulate
from .recurrent import pack
from .contact_events import event_labels,EVENT_NAMES
from .architecture_data import digest

STRATA=['approach','ramp','support','repeated','near_miss','oblique','competing','separating']


def scene_for(seed,index):
    rng=np.random.default_rng(seed);kind=STRATA[index%8]
    if kind in ['approach','ramp']:return generate_scene(seed)
    floor=Segment(id='floor',a=(.1,.18),b=(5.9,.18));segments=(floor,)
    ball=Ball();bumper=Bumper(position=(4.8,2.7))
    if kind in ['support','repeated']:
        rest=rng.uniform(0,.15) if kind=='support' else rng.uniform(.4,.65)
        ball=Ball(position=(rng.uniform(.7,1.6),rng.uniform(.45,1.1)),velocity=(rng.uniform(-.3,.3),0),material=Material(restitution=rest,friction=.6))
    elif kind=='near_miss':
        bumper=Bumper(position=(3.,2.));ball=Ball(position=(3.44+rng.uniform(-.05,.05),3.1),velocity=(rng.uniform(-.1,.1),0))
    elif kind=='oblique':
        bumper=Bumper(position=(3.,2.));ball=Ball(position=(rng.uniform(2.4,2.8),3.1),velocity=(rng.uniform(.6,1.6),0),angular_velocity=rng.uniform(-5,5))
    elif kind=='competing':
        segments=(floor,Segment(id='ramp',a=(3.,.18),b=(4.8,1.4)))
        ball=Ball(position=(rng.uniform(2.65,3.1),rng.uniform(.8,1.1)),velocity=(rng.uniform(.4,2),rng.uniform(-1,0)))
    else:
        bumper=Bumper(position=(3.,2.));ball=Ball(position=(3.45,2.),velocity=(rng.uniform(1,3),0))
    return validate_scene(SceneSpec(id=f'episode-{seed}',parent_id=f'episode-parent-{seed}',seed=seed,family='authored',ball=ball,bumper=bumper,segments=segments))


def build(root,protocol):
    config=json.loads(protocol.read_text());root.mkdir(parents=True,exist_ok=True)
    for split,(seed0,count) in config['parent_pools'].items():
        folder=root/split;manifest_path=folder/'manifest.json'
        if manifest_path.exists():
            previous=json.loads(manifest_path.read_text())
            if previous['protocol_sha256']!=digest(protocol) or previous['sha256']!=digest(folder/'paired.npz'):raise ValueError('data changed')
            continue
        folder.mkdir(exist_ok=True);arrays={k:[] for k in ['states','contexts','contacts','event_labels','active_start','event_times','event_time_mask','event_counts']};records=[];audits=[]
        for i in range(count):
            seed=seed0+i;scene=scene_for(seed,i);grid,valid,_=candidates(scene,9);edits=np.where(valid,grid,0)
            row={k:[] for k in arrays};events=[]
            for b,edit in enumerate(edits):
                tr=simulate(scene,float(edit),record_responses=True);target=event_labels(tr);events.append(target.pop('ordered_events'))
                contacts=np.zeros((120,5),np.float32);contacts[:,:len(tr.object_ids)]=tr.contacts
                context=pack(apply_edit(scene,EditSpec(displacement=float(edit))))[1]
                values=dict(states=tr.states,contexts=context,contacts=contacts,**target)
                for k in arrays:row[k].append(values[k])
                if i<8 and b in [0,4,8]:
                    fine=simulate(scene,float(edit),3840)
                    delta=float(np.linalg.norm(tr.states[:,:2]-fine.states[:,:2],axis=-1).max())
                    audits.append(dict(parent_id=scene.parent_id,edit_m=float(edit),max_position_delta_m=delta,event_count=len(tr.events),fine_event_count=len(fine.events),sensitive=delta>.05 or len(tr.events)!=len(fine.events)))
            for k in arrays:arrays[k].append(row[k])
            records.append(dict(scene=scene.model_dump(),stratum=STRATA[i%8],edits_m=edits.tolist(),valid=valid.tolist(),object_ids=tr.object_ids,events=events))
            if i%16==0:print(split,seed,flush=True)
        arrays={k:np.asarray(v,np.float32) for k,v in arrays.items()};np.savez_compressed(folder/'paired.npz',**arrays)
        support={name:dict(intervals=int(arrays['event_labels'][...,k].sum()),parents=int((arrays['event_labels'][...,k].sum((1,2,3))>0).sum())) for k,name in enumerate(EVENT_NAMES)}
        manifest_path.write_text(json.dumps(dict(records=records,labels=EVENT_NAMES,support=support,numerical_audit=audits,protocol_sha256=digest(protocol),sha256=digest(folder/'paired.npz')),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('data/episodes'));p.add_argument('--protocol',type=Path,default=Path('configs/contact-episodes.json'));a=p.parse_args();build(a.root,a.protocol)
