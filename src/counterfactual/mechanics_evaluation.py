"""Common validation/development gates; no test-selected checkpoint or threshold."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .models import load_predictor
from .contact_suite import match_events
from .schemas import SceneSpec
from .geometry import apply_edit
from .schemas import EditSpec


def stats(a):
    a=np.array(a,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),nonzero=int((a>1e-5).sum())) if len(a) else dict(n=0,mean=None,median=None,p95=None,nonzero=0)


def classify(a,p):
    a=np.array(a,bool);p=np.array(p,bool)
    tp=int((a&p).sum());fp=int((~a&p).sum());fn=int((a&~p).sum())
    return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn),true_intervals=int(a.sum()),total_intervals=a.size)


def measure(predictor,data,split):
    manifest=json.loads((data/'manifest.json').read_text());arrays=np.load(data/'paired.npz')
    parents=[];all_pre=[];all_null=[];event={name:[[],[]] for name in ['bumper','floor','ramp']}
    timing={name:dict(matched=0,invented=0,missed=0,errors=[]) for name in event}
    for i,r in enumerate(manifest['records']):
        if r['split']!=split:continue
        scene=SceneSpec.model_validate(r['scene']);p,score=predictor.forecast(scene,r['edits_m']);truth=arrays['states'][i];contact=arrays['contacts'][i].astype(bool)
        ids=sorted([scene.bumper.id,*[s.id for s in scene.segments]]);bi=ids.index(scene.bumper.id)
        error=np.linalg.norm(p[:,:,:2]-truth[:,:,:2],axis=-1)/scene.ball.radius
        effect=p[:,:,:2]-p[4:5,:,:2];actual=truth[:,:,:2]-truth[4:5,:,:2]
        nonzero=np.array(r['edits_m'])!=0;pre=[];null=[];post=[]
        for b in range(9):
            hit=np.flatnonzero(contact[b].any(-1))
            if len(hit):post.append(float(error[b,min(120,int(hit[0])+7)]))
            if nonzero[b]:
                hit=np.flatnonzero(contact[[4,b],:,bi].any(0));end=int(hit[0])+1 if len(hit) else 121
                pre.append(float(np.linalg.norm(effect[b,:end],axis=-1).mean()/scene.ball.radius))
                if not len(hit):null.append(float(np.linalg.norm(effect[b],axis=-1).mean()/scene.ball.radius))
        all_pre+=pre;all_null+=null
        for j,obj in enumerate(ids):
            event[obj][0].extend(contact[:,:,j].ravel());event[obj][1].extend((score[:,:,j]>=.5).ravel())
            for b in range(9):
                on=lambda flags:(np.flatnonzero(flags & ~np.r_[False,flags[:-1]])+1)/60
                matched=match_events(on(contact[b,:,j]),on(score[b,:,j]>=.5))
                for k in ['matched','invented','missed']:timing[obj][k]+=matched[k]
                timing[obj]['errors']+=matched['absolute_timing_errors_s']
        parents.append(dict(parent_id=scene.parent_id,family=scene.family,position_mean_radii=float(error.mean()),position_p95_radii=float(np.quantile(error,.95)),difference_radii=float(np.linalg.norm(effect[nonzero]-actual[nonzero],axis=-1).mean()/scene.ball.radius),zero_effect_predictor_error_radii=float(np.linalg.norm(actual[nonzero],axis=-1).mean()/scene.ball.radius),pre=pre,no_effect=null,post_impact_radii=stats(post)))
    return dict(split=split,parents=parents,pre_influence=stats(all_pre),no_effect=stats(all_null),contact={o:classify(*v) for o,v in event.items()},onsets={o:{**{k:v for k,v in t.items() if k!='errors'},'timing_s':stats(t['errors'])} for o,t in timing.items()},families={f:{k:stats([p[k] for p in parents if p['family']==f]) for k in ['position_mean_radii','position_p95_radii','difference_radii']} for f in ['approach','ramp']})


def suite_measure(predictor,suite):
    rows=[]
    for r in json.loads(suite.read_text())['records']:
        if not r['valid']:continue
        scene=SceneSpec.model_validate(r['scene']);p,score=predictor.forecast(scene,[r['edit_m']]);truth=np.array(r['states']);objects={}
        for j,obj in enumerate(r['object_ids']):
            flags=score[0,:,j]>=.5
            predicted=(np.flatnonzero(flags & ~np.r_[False,flags[:-1]])+1)/60
            actual=[e['time'] for e in r['onsets'] if obj in e['objects']]
            objects[obj]=match_events(actual,predicted)
        rows.append(dict(fixture=scene.id,edit_m=r['edit_m'],numerically_sensitive=r['numerically_sensitive'],position_mean_radii=float(np.linalg.norm(p[0,:,:2]-truth[:,:2],axis=-1).mean()/scene.ball.radius),events=objects))
    return rows


def run(checkpoint,output,data=Path('data/pilot')):
    torch.set_num_threads(4);started=perf_counter();p=load_predictor(checkpoint)
    report=dict(model=p.identity,validation=measure(p,data,'validation'),development=measure(p,data,'test'),suite=suite_measure(p,Path('artifacts/contact-suite/suite.json')),seconds=perf_counter()-started)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2));print(json.dumps({'checkpoint':str(checkpoint),'validation_pre':report['validation']['pre_influence'],'validation_no_effect':report['validation']['no_effect']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/mechanics'));a=p.parse_args()
    for cp in sorted(a.root.glob('*/seed-*/model.pt')):
        out=cp.parent/'diagnostics.json'
        if not out.exists():run(cp,out)
    for name,cp in [('E0','artifacts/pilot/baseline.pt'),('A','artifacts/recurrent-a/recurrent.pt')]:
        out=a.root/'references'/f'{name}.json'
        if not out.exists():run(Path(cp),out)
