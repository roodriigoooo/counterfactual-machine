"""T2/T3 matched-update mechanics × exposure experiment, with local control."""
import argparse,hashlib,json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from .contact_mechanics import MechanicsModel
from .recurrent import pack
from .geometry import apply_edit
from .schemas import SceneSpec,EditSpec

CELLS={'soft-window':('soft','window','gru'),'soft-release':('soft','release','gru'),
       'swept-window':('swept','window','gru'),'swept-release':('swept','release','gru'),
       'swept-local-release':('swept','release','local')}


def dataset(path):
    manifest=json.loads((path/'manifest.json').read_text());arrays=np.load(path/'paired.npz')
    if hashlib.sha256((path/'paired.npz').read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('dataset hash mismatch')
    context=[];bumper=[]
    for r in manifest['records']:
        s=SceneSpec.model_validate(r['scene']);bumper.append(sorted([s.bumper.id,*[o.id for o in s.segments]]).index(s.bumper.id))
        context.append([pack(apply_edit(s,EditSpec(object_id=s.bumper.id,displacement=u)))[1] for u in r['edits_m']])
    return manifest,arrays,torch.tensor(np.array(context)),bumper


def train(path,out,cell,seed,updates):
    torch.set_num_threads(4);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    mechanics,exposure,memory=CELLS[cell];out=out/cell/f'seed-{seed}'
    if (out/'model.pt').exists():raise ValueError(f'refusing to overwrite {out}')
    out.mkdir(parents=True,exist_ok=True)
    manifest,arrays,contexts,bumper=dataset(path)
    states=torch.tensor(arrays['states']);contacts=torch.tensor(arrays['contacts'])
    ids=np.flatnonzero(arrays['split']==0);val=np.flatnonzero(arrays['split']==1)
    model=MechanicsModel(mechanics,memory);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    best=float('inf');history=[];started=perf_counter();processed=0
    for update in range(1,updates+1):
        parents=rng.choice(ids,8);branches=rng.choice([0,1,2,3,5,6,7,8],8)
        # Generate the same sampled windows in all cells for a given seed.
        starts=[]
        for p,b in zip(parents,branches):
            hit=arrays['contacts'][p,b].max(-1)>0
            # Half contact, half strictly non-contact starting intervals when available.
            pool=np.flatnonzero(hit if rng.random()<.5 else ~hit)
            start=int(rng.choice(pool)) if len(pool) else 0
            starts.append(min(max(0,start-int(rng.integers(0,8))),104))
        ps=np.repeat(parents,2);bs=np.column_stack([np.full(8,4),branches]).ravel()
        starts=np.repeat(starts,2) if exposure=='window' else np.zeros(16,dtype=int)
        horizon=16 if exposure=='window' else 120
        truth=torch.stack([states[p,b,t:t+horizon+1] for p,b,t in zip(ps,bs,starts)])
        label=torch.stack([contacts[p,b,t:t+horizon] for p,b,t in zip(ps,bs,starts)])
        context=contexts[ps,bs];prediction,logits=model(truth[:,0],context,horizon)
        scale=prediction.new_tensor([.12,.12,3,3,10,20]);error=(prediction-truth)/scale
        physical=error.square().mean();pair=error.reshape(8,2,horizon+1,6)
        difference=(pair[:,1]-pair[:,0]).square().mean()
        influence=[]
        for i,(p,b) in enumerate(zip(parents,branches)):
            actual=arrays['contacts'][p,[4,b],:,bumper[p]].any(0)
            # states up to start of first contact interval are pre-influence.
            first=np.flatnonzero(actual);end=int(first[0]) if len(first) else 120
            influence.append(np.arange(horizon+1)+starts[i*2]<=end)
        pre=torch.tensor(np.array(influence),dtype=prediction.dtype)
        pred_pair=(prediction/scale).reshape(8,2,horizon+1,6)
        premature=((pred_pair[:,1]-pred_pair[:,0]).square().mean(-1)*pre).sum()/pre.sum().clamp_min(1)
        mask=context[:,None,:,9]
        event=(F.binary_cross_entropy_with_logits(logits,label,reduction='none')*mask).sum()/(mask.sum()*horizon)
        dv=prediction[:,1:,2:4]-prediction[:,:-1,2:4]-prediction.new_tensor([0,-9.81/60])
        noncontact=(label.sum(-1)==0).float()
        unsupported=(dv.square().sum(-1)*noncontact).sum()/noncontact.sum().clamp_min(1)
        loss=physical+.5*difference+premature+.1*event+.1*unsupported
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        processed+=16*horizon
        if update==1 or update%50==0 or update==updates:
            with torch.inference_mode():
                all_errors=[]
                for block in np.array_split(val,4):
                    p,_=model(states[block,:,0].flatten(0,1),contexts[block].flatten(0,1))
                    all_errors.append((p[:,:,:2]-states[block].flatten(0,1)[:,:,:2]).square().sum(-1).flatten())
                score=float(torch.cat(all_errors).mean().sqrt())
            row=dict(update=update,loss=float(loss.detach()),physical=float(physical.detach()),difference=float(difference.detach()),pre_influence=float(premature.detach()),contact=float(event.detach()),unsupported=float(unsupported.detach()),validation_vector_rmse_m=score,seconds=perf_counter()-started)
            history.append(row);print(json.dumps(dict(cell=cell,seed=seed,**row)),flush=True)
            if score<best:
                best=score;torch.save(dict(model=model.state_dict(),name='Contact mechanics experiment',mechanics=mechanics,memory=memory,exposure=exposure,seed=seed,epoch=update,dataset_sha256=manifest['sha256']),out/'model.pt')
    (out/'training.json').write_text(json.dumps(dict(cell=cell,seed=seed,updates=updates,parameters=sum(p.numel() for p in model.parameters()),processed_training_intervals=processed,validation_parents=val.tolist(),training_parents=ids.tolist(),dataset_sha256=manifest['sha256'],seconds=perf_counter()-started,history=history),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/pilot'));p.add_argument('--output',type=Path,default=Path('artifacts/mechanics'));p.add_argument('--cells',nargs='+',choices=list(CELLS),default=list(CELLS));p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--updates',type=int,default=200)
    a=p.parse_args()
    for seed in a.seeds:
        for cell in a.cells:train(a.data,a.output,cell,seed,a.updates)
