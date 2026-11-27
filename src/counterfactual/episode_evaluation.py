"""Event-head value beyond masking, with exact and self-generated episode state."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .architecture_training import load_data
from .architecture_data import digest
from .episode_training import cached_paths
from .episode_model import EpisodeModel
from .contact_events import EVENT_NAMES
from .phase_evaluation import binary
from .contact_suite import match_events
from .architecture import conditions


def event_metrics(prob,timing,labels,time_targets,time_mask,mask,records):
    metrics={};n=len(records)
    for j,name in enumerate(EVENT_NAMES):
        truth=labels[...,j]>.5;selected=prob[...,j]>=.5
        metrics[name]=binary(truth[mask],selected[mask])
        metrics[name]['parents']=int((truth&mask).reshape(n,-1).any(1).sum())
    timing_result={}
    for j,name in enumerate(['begin','separate']):
        use=(time_mask[...,j]>.5)&mask;errors=abs(timing[...,j]-time_targets[...,j])[use]/60
        matches=dict(matched=0,invented=0,missed=0);deltas=[]
        for i,record in enumerate(records):
            for b in range(9):
                for edge,obj in enumerate(record['object_ids']):
                    truth=[e['time_s'] for e in record['events'][b] if e['object']==obj and e['phase']==name]
                    indices=np.flatnonzero(prob[i,b,:,edge,1+j]>=.5)
                    predicted=(indices+timing[i,b,indices,edge,j])/60
                    found=match_events(truth,predicted)
                    for k in matches:matches[k]+=found[k]
                    deltas+=found['absolute_timing_errors_s']
        tp,fp,fn=matches['matched'],matches['invented'],matches['missed']
        timing_result[name]=dict(**matches,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
            f1=2*tp/max(1,2*tp+fp+fn),matched_time_mae_ms=float(np.mean(deltas)*1000) if deltas else None,
            conditional_first_event_mae_ms=float(errors.mean()*1000) if len(errors) else None,
            conditional_first_event_count=len(errors),tolerance_s=1/30)
    return dict(labels=metrics,events=timing_result)


@torch.inference_mode()
def evaluate(cp,data_folder):
    torch.set_num_threads(4);manifest,data=load_data(data_folder);payload=torch.load(cp,map_location='cpu',weights_only=True)
    model=EpisodeModel(payload['cell']);model.load_state_dict(payload['model']);model.eval()
    shape=data['event_labels'].shape;n=len(data['states']);c=data['contexts'].flatten(0,1);paths=cached_paths(data_folder,data)
    mask=data['contexts'][:,:,None,:,9].expand(*shape[:-1]).bool().numpy();labels=data['event_labels'].numpy();target_times=data['event_times'].numpy();tm=data['event_time_mask'].numpy()
    record=dict(checkpoint_sha256=digest(cp),dataset_sha256=manifest['sha256'],cell=payload['cell'],seed=payload['seed'],update=payload['update'],metrics={})
    for mode in ['exact_oracle_history','exact_self_history','release']:
        states=(paths if mode=='release' else data['states']).flatten(0,1)
        if mode=='exact_oracle_history':
            logits=[];times=[]
            for block in torch.arange(len(states)).split(24):
                context=c[block,None].expand(-1,120,-1,-1).flatten(0,1)
                active=data['active_start'].flatten(0,1)[block].flatten(0,1)
                a,b=model.readout(states[block,:-1].flatten(0,1),context,active)
                logits.append(a.reshape(len(block),120,5,6));times.append(b.reshape(len(block),120,5,2))
            logits=torch.cat(logits);times=torch.cat(times)
        else:logits,times,_=model.rollout_readout(states,c)
        probability=logits.sigmoid().reshape(*shape).numpy();timing=times.reshape(n,9,120,5,2).numpy()
        record['metrics'][mode]=event_metrics(probability,timing,labels,target_times,tm,mask,manifest['records'])
        # Geometry is an occupancy/impact opportunity control, not a true callback sequence.
        context=c[:,None].expand(-1,120,-1,-1).flatten(0,1);_,geo=conditions(states[:,:-1].flatten(0,1),context)
        eligibility=geo[0].reshape(*shape[:-1]).numpy();admitted=geo[1].reshape(*shape[:-1]).numpy()
        occupancy=labels[...,0]>.5
        record['metrics'][mode]['occupancy_controls']=dict(raw=binary(occupancy[mask],probability[...,0][mask]>=.5),
            masked=binary(occupancy[mask],(probability[...,0][mask]>=.5)&eligibility[mask]),
            always_positive_masked=binary(occupancy[mask],eligibility[mask]),earliest_geometry=binary(occupancy[mask],admitted[mask]))
        record['metrics'][mode]['strata']={}
        for stratum in sorted(set(r['stratum'] for r in manifest['records'])):
            idx=np.array([r['stratum']==stratum for r in manifest['records']]);use=mask[idx]
            record['metrics'][mode]['strata'][stratum]={name:binary(labels[idx,...,j][use]>.5,probability[idx,...,j][use]>=.5) for j,name in enumerate(EVENT_NAMES)}
    record['limitations']=dict(event_readout_max_per_object_interval=1,actual_intervals_exceeding_readout=int((data['event_counts']>1).any(-1).sum()),physics='All cells use identical unchanged analytic trajectories; no learned physical improvement is claimed')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/episodes'));p.add_argument('--data',type=Path,default=Path('data/episodes/validation'));a=p.parse_args()
    for cp in sorted(a.root.glob('*/seed-*/update-*.pt')):
        out=cp.with_suffix('.json')
        if out.exists():
            if json.loads(out.read_text())['checkpoint_sha256']!=digest(cp):raise ValueError('stale event evaluation')
            continue
        record=evaluate(cp,a.data);out.write_text(json.dumps(record,indent=2));print(cp,record['metrics']['release']['events']['begin'],flush=True)
