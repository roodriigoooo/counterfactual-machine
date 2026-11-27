"""Event identifiability, then causal episode history on analytic free releases."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from .architecture_training import load_data
from .architecture_data import digest
from .episode_model import EpisodeModel,analytic_paths


def event_loss(logits,timing,labels,target_times,time_mask,mask):
    ce=F.binary_cross_entropy_with_logits(logits,labels,reduction='none');parts=[]
    for j in range(labels.shape[-1]):
        classes=[]
        for positive in [False,True]:
            eligible=mask.bool()&((labels[...,j]>.5)==positive)
            if eligible.any():classes.append(ce[...,j][eligible].mean())
        if classes:parts.append(torch.stack(classes).mean())
    classification=torch.stack(parts).mean()
    eligible=time_mask*mask[...,None]
    timing_error=((timing-target_times).square()*eligible).sum()/eligible.sum().clamp_min(1)
    return classification+.2*timing_error,classification,timing_error


def cached_paths(folder,data):
    path=folder/'analytic-paths.npz'
    if path.exists():
        with np.load(path) as a:
            if str(a['dataset_sha256'])!=digest(folder/'paired.npz'):raise ValueError('stale analytic cache')
            return torch.from_numpy(a['states'].copy())
    initial=data['states'][:,:,0].flatten(0,1);c=data['contexts'].flatten(0,1);paths=[]
    for idx in torch.arange(len(initial)).split(64):paths.append(analytic_paths(initial[idx],c[idx]))
    result=torch.cat(paths).reshape(*data['states'].shape)
    np.savez_compressed(path,states=result.numpy(),dataset_sha256=digest(folder/'paired.npz'))
    return result


def train(root,data_root,protocol,cell,seed):
    out=root/cell/f'seed-{seed}'
    if (out/'training.json').exists():raise ValueError(f'refusing to overwrite {out}')
    config=json.loads(protocol.read_text());manifest,data=load_data(data_root/'train')
    if manifest['protocol_sha256']!=digest(protocol):raise ValueError('protocol mismatch')
    root.mkdir(parents=True,exist_ok=True);frozen=root/'protocol.json'
    if frozen.exists() and frozen.read_bytes()!=protocol.read_bytes():raise ValueError('protocol changed')
    if not frozen.exists():frozen.write_bytes(protocol.read_bytes())
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=EpisodeModel(cell);optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate']);paths=cached_paths(data_root/'train',data)
    history=[];started=perf_counter();n=config['batch_parents']
    for update in range(1,301):
        parents=rng.choice(len(data['states']),n,replace=False);branches=rng.integers(0,9,n);c=data['contexts'][parents,branches]
        labels=data['event_labels'][parents,branches];timing=data['event_times'][parents,branches];time_mask=data['event_time_mask'][parents,branches]
        mask=c[:,None,:,9].expand(-1,120,-1)
        if update<=200:
            state=data['states'][parents,branches,:-1].flatten(0,1);context=c[:,None].expand(-1,120,-1,-1).flatten(0,1)
            active=data['active_start'][parents,branches].flatten(0,1)
            logits,predicted_times=model.readout(state,context,active)
            logits=logits.reshape(n,120,5,6);predicted_times=predicted_times.reshape(n,120,5,2)
        else:logits,predicted_times,_=model.rollout_readout(paths[parents,branches],c)
        loss,ce,te=event_loss(logits,predicted_times,labels,timing,time_mask,mask)
        if not torch.isfinite(loss):raise FloatingPointError('event loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
        if update==1 or update%50==0:
            row=dict(update=update,classification=float(ce.detach()),timing=float(te.detach()),seconds=perf_counter()-started);history.append(row);print(cell,seed,json.dumps(row),flush=True)
        if update in [200,300]:
            torch.save(dict(model=model.state_dict(),cell=cell,seed=seed,update=update,protocol_sha256=digest(protocol),dataset_sha256=manifest['sha256']),out/f'update-{update}.pt')
    (out/'training.json').write_text(json.dumps(dict(cell=cell,seed=seed,history=history,seconds=perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),physical_weights_trained=False,exact_state_intervals=200*n*120,release_intervals=100*n*120,protocol_sha256=digest(protocol)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/episodes'));p.add_argument('--data',type=Path,default=Path('data/episodes'));p.add_argument('--protocol',type=Path,default=Path('configs/contact-episodes.json'));p.add_argument('--cells',nargs='+',default=['E0','E1','E2']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);a=p.parse_args()
    for seed in a.seeds:
        for cell in a.cells:train(a.root,a.data,a.protocol,cell,seed)
