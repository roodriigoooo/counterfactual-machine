"""Exact-state diagnostic: unsupported mechanical changes, not deployable results."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .models import load_predictor
from .mechanics_experiment import dataset


def audit(checkpoint,data):
    predictor=load_predictor(checkpoint);model=predictor.model
    manifest,arrays,contexts,_=dataset(data)
    ids=np.flatnonzero(arrays['split']==1)
    truth=torch.tensor(arrays['states'][ids]).flatten(0,1)
    labels=arrays['contacts'][ids].reshape(-1,120,5).any(-1)
    nearby=labels.copy();nearby[:,1:] |= labels[:,:-1];nearby[:,:-1] |= labels[:,1:]
    c=contexts[ids].flatten(0,1);encoded=model.encode(c);h=torch.zeros(len(c),5,32)
    dv=[];dp=[];scores=[]
    with torch.inference_mode():
        for t in range(120):
            y,h,logit=model.step(truth[:,t],c,encoded,h)
            dv.append((y[:,2:4]-truth[:,t,2:4]-y.new_tensor([0,-9.81/60])).norm(dim=-1).numpy())
            prior=truth[:,t,:2]+truth[:,t,2:4]/60+y.new_tensor([0,-9.81])*(1/60*(1/60-1/1920)/2)
            dp.append((y[:,:2]-prior).norm(dim=-1).numpy());scores.append(logit.sigmoid().max(-1).values.numpy())
    dv=np.stack(dv,1);dp=np.stack(dp,1);scores=np.stack(scores,1)
    rows={}
    for name,mask in [('non_contact',~labels),('away_from_contact',~nearby)]:
        rows[name]=dict(intervals=int(mask.sum()),velocity_mean_mps=float(dv[mask].mean()),velocity_p95_mps=float(np.quantile(dv[mask],.95)),velocity_above_005=int(((dv>.05)&mask).sum()),weak_score_strong_response=int(((dv>.05)&(scores<.5)&mask).sum()),position_correction_mean_m=float(dp[mask].mean()))
    return dict(model=predictor.identity,scope='validation exact-state roll-ins, true history in memory, all-object total response; not per-edge counts or free-running performance',results=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/mechanics'));a=p.parse_args();torch.set_num_threads(4)
    records={}
    for cp in sorted(a.root.glob('*/seed-0/model.pt')):
        records[str(cp.parent.relative_to(a.root))]=audit(cp,Path('data/pilot'))
    records['frozen-A']=audit(Path('artifacts/recurrent-a/recurrent.pt'),Path('data/pilot'))
    (a.root/'exact-state-audit.json').write_text(json.dumps(records,indent=2))
