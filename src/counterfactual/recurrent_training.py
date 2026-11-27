"""Staged short-window training; validation free runs select checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from .recurrent import InteractionRollout, pack
from .geometry import apply_edit
from .schemas import SceneSpec, EditSpec


def train(data, output, updates=800, seed=0):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    manifest = json.loads((data/'manifest.json').read_text())
    if hashlib.sha256((data/'paired.npz').read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('dataset hash mismatch')
    arrays = np.load(data/'paired.npz')
    states = torch.tensor(arrays['states'])
    contacts = torch.tensor(arrays['contacts'])
    contexts = []
    for record in manifest['records']:
        scene = SceneSpec.model_validate(record['scene'])
        contexts.append([pack(apply_edit(scene, EditSpec(object_id=scene.bumper.id, displacement=u)))[1] for u in record['edits_m']])
    contexts = torch.tensor(np.array(contexts))
    train_ids = np.flatnonzero(arrays['split'] == 0)
    val_ids = np.flatnonzero(arrays['split'] == 1)
    model = InteractionRollout()
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    history = []; best = float('inf'); started = perf_counter()
    for update in range(1, updates+1):
        horizon = [4, 8, 16, 32][min(3, (update-1)*4//updates)]
        parents = rng.choice(train_ids, 16)
        branches = rng.integers(0, 9, 16)
        starts = []
        for p, b in zip(parents, branches):
            event = np.flatnonzero(arrays['contacts'][p,b].max(-1) > 0)
            start = max(0, int(rng.choice(event))-int(rng.integers(1, horizon+1))) if len(event) and rng.random() < .75 else int(rng.integers(0,121-horizon))
            starts.append(min(start, 120-horizon))
        # Aligned original/edited windows: paired loss does not receive future inputs.
        ps = np.repeat(parents, 2); bs = np.column_stack([np.full(16,4), branches]).ravel()
        ts = np.repeat(starts, 2)
        truth = torch.stack([states[p,b,t:t+horizon+1] for p,b,t in zip(ps,bs,ts)])
        labels = torch.stack([contacts[p,b,t:t+horizon] for p,b,t in zip(ps,bs,ts)])
        context = contexts[ps,bs]
        prediction, logits = model(truth[:,0], context, horizon)
        scale = torch.tensor([.12,.12,3.,3.,10.,20.])
        error = (prediction-truth)/scale
        physical = error.square().mean()
        pair = error.reshape(16,2,horizon+1,6)
        difference = (pair[:,1]-pair[:,0]).square().mean()
        mask = context[:,None,:,9]
        contact_loss = (F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*mask).sum()/(mask.sum()*horizon)
        loss = physical+.5*difference+.1*contact_loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2); opt.step()
        if update == 1 or update % 100 == 0 or update == updates:
            with torch.inference_mode():
                errors=[]
                for ids in np.array_split(val_ids,4):
                    pred,_=model(states[ids,:,0].flatten(0,1),contexts[ids].flatten(0,1))
                    errors.append(float((pred[:,:,:2]-states[ids].flatten(0,1)[:,:,:2]).square().mean().sqrt()))
            score=float(np.mean(errors))
            row=dict(update=update,horizon=horizon,loss=float(loss.detach()),physical=float(physical.detach()),paired=float(difference.detach()),contact=float(contact_loss.detach()),validation_coordinate_rmse_m=score,seconds=perf_counter()-started)
            history.append(row); print(json.dumps(row),flush=True)
            if score < best:
                best=score
                torch.save(dict(model=model.state_dict(),name='A recurrent interaction prototype',seed=seed,epoch=update,dataset_sha256=manifest['sha256']),output/'recurrent.pt')
    record=dict(model='A recurrent interaction prototype',seed=seed,updates=updates,parameters=sum(p.numel() for p in model.parameters()),dataset_sha256=manifest['sha256'],best_validation_coordinate_rmse_m=best,seconds=perf_counter()-started,history=history,
                limitations=['Window starts use true states during training; complete inference is free-running from release.', 'Memory is reset at training window starts; burn-in remains pending.', 'Paired physical loss includes no-effect targets; explicit event-timed pre-influence weighting pending.', 'Timing fraction is trained through motion; explicit substep timing and phase labels pending.'])
    (output/'training.json').write_text(json.dumps(record,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/pilot'));p.add_argument('--output',type=Path,default=Path('artifacts/recurrent-a'));p.add_argument('--updates',type=int,default=800);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();train(a.data,a.output,a.updates,a.seed)
