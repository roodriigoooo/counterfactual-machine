"""Three-seed factorial training; checkpoint selection never reads final parents."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from .architecture import ArchitectureModel
from .architecture_data import digest


def load_data(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['sha256']!=digest(folder/'paired.npz'):raise ValueError('dataset hash mismatch')
    with np.load(folder/'paired.npz') as a:
        return manifest,{k:torch.from_numpy(a[k].copy()) for k in a.files}


def paired_rollout(model, initial, c0, cu, auxiliary=True):
    if model.variant in 'BD':return model.paired(initial,c0,cu,auxiliary=auxiliary)
    output=model.rollout(torch.cat([initial,initial]),torch.cat([c0,cu]),auxiliary=auxiliary)
    n=len(initial)
    return ({k:v[:n] for k,v in output.items()},{k:v[n:] for k,v in output.items()})


def variance_covariance(z):
    z=z-z.mean(0)
    variance=F.relu(1-torch.sqrt(z.var(0,unbiased=False)+1e-4)).mean()
    covariance=z.T@z/max(1,len(z)-1)
    off=covariance-torch.diag_embed(covariance.diag())
    return variance,off.square().sum()/z.shape[1]


def embedding_diagnostics(z):
    z=z-z.mean(0)
    singular=torch.linalg.svdvals(z.double())
    probability=singular/singular.sum().clamp_min(1e-12)
    return dict(mean_std=float(z.std(0,unbiased=False).mean()),
                effective_rank=float((-(probability*probability.clamp_min(1e-12).log()).sum()).exp()))


@torch.inference_mode()
def validation(model,data):
    errors=[]
    for block in torch.arange(len(data['states'])).split(8):
        context=data['contexts'][block].flatten(0,1);initial=data['states'][block,:,0].flatten(0,1)
        reference=None
        if model.variant in 'BD':
            ref=model.rollout(data['states'][block,4,0],data['contexts'][block,4])
            reference={k:v.repeat_interleave(9,0) for k,v in ref.items()}
        prediction=model.rollout(initial,context,reference=reference)['states']
        errors.append((prediction[...,:2]-data['states'][block].flatten(0,1)[...,:2]).square().sum(-1).flatten())
    c=data['contexts'][:,4];states=data['states'][:,4,60]
    _,z,_=model.representation(states,c)
    z=(z*c[...,9:10]).sum(1)/c[...,9:10].sum(1)
    return float(torch.cat(errors).mean().sqrt()),embedding_diagnostics(z)


def train(data_root,root,protocol_path,variant,seed):
    config=json.loads(protocol_path.read_text());out=root/variant/f'seed-{seed}'
    if (out/'model.pt').exists():raise ValueError(f'refusing to overwrite {out}')
    out.mkdir(parents=True,exist_ok=True)
    manifest,data=load_data(data_root/'train');_,val=load_data(data_root/'validation')
    if manifest['protocol_sha256']!=digest(protocol_path):raise ValueError('protocol mismatch')
    torch.manual_seed(seed);torch.set_num_threads(4);rng=np.random.default_rng(seed)
    model=ArchitectureModel(variant);target=deepcopy(model.representation).requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
    weights=config['shared_losses'];joint_weights=config['embedding_losses'];joint=variant in 'CD'
    n=config['batch_parents'];scale=torch.tensor([.12,.12,3,3,10,20]);response_scale=torch.tensor([5.,5.,20.])
    steps=torch.tensor(config['embedding_times']);best=float('inf');history=[];started=perf_counter()
    for update in range(1,config['updates']+1):
        parents=rng.choice(len(data['states']),n,replace=False);branches=rng.choice([0,1,2,3,5,6,7,8],n)
        original,edited=paired_rollout(model,data['states'][parents,4,0],data['contexts'][parents,4],data['contexts'][parents,branches])
        pred=torch.stack([original['states'],edited['states']],1)
        truth=torch.stack([data['states'][parents,4],data['states'][parents,branches]],1)
        context=torch.stack([data['contexts'][parents,4],data['contexts'][parents,branches]],1)
        labels={k:torch.stack([data[k][parents,4],data[k][parents,branches]],1) for k in ['contacts','phase','onset','onset_mask','response']}
        mask=context[:,:,None,:,9].expand(-1,-1,120,-1)
        error=(pred-truth)/scale
        losses=dict(physical=error.square().mean(),physical_difference=(error[:,1]-error[:,0]).square().mean())
        # Canonical objects put bumper first in both generated families.
        touched=labels['contacts'][... ,0].any(1).bool()
        first=torch.where(touched,torch.arange(120),torch.tensor(120)).amin(1)
        before=torch.arange(121)[None,:]<=first[:,None]
        pair=(pred[:,1]-pred[:,0])/scale
        losses['physical_pre_influence']=(pair.square().mean(-1)*before).sum()/before.sum().clamp_min(1)
        phase=torch.stack([original['phase'],edited['phase']],1)
        ce=F.cross_entropy(phase.flatten(0,3),labels['phase'].long().flatten(),weight=torch.tensor([.05,1.,1.,.5]),reduction='none').reshape_as(mask)
        losses['phase']=(ce*mask).sum()/mask.sum()
        onset=torch.stack([original['onset'],edited['onset']],1)
        onset_mask=labels['onset_mask']*mask
        losses['onset_time']=((onset-labels['onset']).square()*onset_mask).sum()/onset_mask.sum().clamp_min(1)
        response=torch.stack([original['response'],edited['response']],1)
        contact_mask=labels['contacts']*mask
        losses['local_response']=(((response-labels['response'])/response_scale).square().mean(-1)*contact_mask).sum()/contact_mask.sum().clamp_min(1)
        dv=pred[:,:,1:,2:4]-pred[:,:,:-1,2:4]-pred.new_tensor([0,-9.81/60])
        absent=labels['contacts'].sum(-1)==0
        losses['unsupported_velocity']=(dv.square().sum(-1)*absent).sum()/absent.sum().clamp_min(1)
        embedding_losses={}
        if joint:
            future=torch.stack([original['future'],edited['future']],1)[:,:,steps]
            selected_truth=truth[:,:,steps+1].flatten(0,2)
            selected_context=context[:,:,None].expand(-1,-1,len(steps),-1,-1).flatten(0,2)
            with torch.no_grad():_,tz,_=target(selected_truth,selected_context)
            tz=tz.reshape(n,2,len(steps),5,16)
            embmask=context[:,:,None,:,9:10]
            embedding_losses['future']=((future-tz).square()*embmask).sum()/(embmask.sum()*len(steps)*16)
            embedding_losses['paired_future_difference']=(((future[:,1]-future[:,0])-(tz[:,1]-tz[:,0])).square()*embmask[:,0]).sum()/(embmask[:,0].sum()*len(steps)*16)
            # One original future per independent parent, not correlated branch/time duplicates.
            at=int(steps[(update-1)%len(steps)])+1
            _,oz,_=model.representation(truth[:,0,at],context[:,0])
            pooled=oz.sum(1)/context[:,0,:,9:10].sum(1)
            embedding_losses['variance'],embedding_losses['covariance']=variance_covariance(pooled)
        loss=sum(weights[k]*v for k,v in losses.items())+sum(joint_weights[k]*v for k,v in embedding_losses.items())
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite {variant}/{seed}/{update}')
        gradients={}
        if update in [1,config['updates']]:
            for group,term in [('physical',losses['physical']),('phase',weights['phase']*losses['phase']),('embedding',sum(joint_weights[k]*v for k,v in embedding_losses.items()))]:
                if torch.is_tensor(term):
                    g=torch.autograd.grad(term,model.representation.edge[0].weight,retain_graph=True,allow_unused=True)[0]
                    gradients[group]=float(g.norm()) if g is not None else 0.
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
        if joint:
            with torch.no_grad():
                for ema,online in zip(target.parameters(),model.representation.parameters()):ema.lerp_(online,1-config['ema_decay'])
        if update==1 or update%50==0 or update==config['updates']:
            score,rank=validation(model,val)
            row=dict(update=update,loss=float(loss.detach()),validation_vector_rmse_m=score,embedding=rank,
                     losses={k:float(v.detach()) for k,v in {**losses,**embedding_losses}.items()},gradient_norms=gradients,seconds=perf_counter()-started)
            history.append(row);print(json.dumps(dict(variant=variant,seed=seed,**row)),flush=True)
            if score<best:
                best=score
                torch.save(dict(model=model.state_dict(),ema=target.state_dict() if joint else {},name='Structured architecture experiment',
                                variant=variant,seed=seed,epoch=update,dataset_sha256=manifest['sha256'],protocol_sha256=digest(protocol_path)),out/'model.pt')
    inactive=sum(p.numel() for module in [model.future,model.representation.dynamic] for p in module.parameters()) if not joint else 0
    (out/'training.json').write_text(json.dumps(dict(variant=variant,seed=seed,updates=config['updates'],parameters=sum(p.numel() for p in model.parameters()),inactive_auxiliary_parameters=inactive,
       processed_training_intervals=config['updates']*n*2*120,seconds=perf_counter()-started,history=history,dataset_sha256=manifest['sha256'],protocol_sha256=digest(protocol_path)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/architecture'));p.add_argument('--root',type=Path,default=Path('artifacts/architecture'));p.add_argument('--protocol',type=Path,default=Path('configs/architecture-experiment.json'));p.add_argument('--variants',nargs='+',default=list('ABCD'));p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);a=p.parse_args()
    for seed in a.seeds:
        for variant in a.variants:train(a.data,a.root,a.protocol,variant,seed)
