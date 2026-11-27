"""A-only loss-scale/input-alignment diagnosis; preserve the factorial baseline."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from .architecture import ArchitectureModel
from .architecture_data import digest
from .architecture_training import load_data, paired_rollout, validation

CLASS_WEIGHT = torch.tensor([.05, 1., 1., .5])


def phase_objective(logits, labels, mask, normalization='count'):
    ce = F.cross_entropy(logits.reshape(-1, 4), labels.long().flatten(),
                         weight=CLASS_WEIGHT, reduction='none').reshape_as(mask)
    denominator = mask.sum() if normalization == 'count' else (CLASS_WEIGHT[labels.long()]*mask).sum()
    return (ce*mask).sum()/denominator.clamp_min(1e-12)


class BalancedTransitions:
    """Sample a phase, then a supporting parent uniformly, then an edge/time."""
    def __init__(self, data, seed):
        self.data, self.rng = data, np.random.default_rng(seed+72193)
        labels = data['phase'].numpy().astype(int)
        valid = np.broadcast_to(data['contexts'][:,:,None,:,9].numpy()>0, labels.shape)
        self.pools = {k:{p:np.argwhere((labels[p]==k)&valid[p]) for p in range(len(labels))} for k in range(4)}
        self.pools = {k:{p:a for p,a in pool.items() if len(a)} for k,pool in self.pools.items()}

    def sample(self, count):
        states, contexts, labels = [], [], []
        for k,pool in self.pools.items():
            parents = np.array(list(pool))
            for p in self.rng.choice(parents, count//4 + int(k<count%4)):
                b,t,j = pool[p][self.rng.integers(len(pool[p]))]
                states.append(self.data['states'][p,b,t]);contexts.append(self.data['contexts'][p,b,j]);labels.append(k)
        return torch.stack(states),torch.stack(contexts)[:,None],torch.tensor(labels)


def common_losses(original, edited, data, parents, branches):
    pred=torch.stack([original['states'],edited['states']],1)
    truth=torch.stack([data['states'][parents,4],data['states'][parents,branches]],1)
    context=torch.stack([data['contexts'][parents,4],data['contexts'][parents,branches]],1)
    labels={k:torch.stack([data[k][parents,4],data[k][parents,branches]],1) for k in ['contacts','phase','onset','onset_mask','response']}
    mask=context[:,:,None,:,9].expand(-1,-1,120,-1)
    scale=torch.tensor([.12,.12,3,3,10,20]);error=(pred-truth)/scale
    losses=dict(physical=error.square().mean(),physical_difference=(error[:,1]-error[:,0]).square().mean())
    touched=labels['contacts'][...,0].any(1).bool()
    first=torch.where(touched,torch.arange(120),torch.tensor(120)).amin(1)
    before=torch.arange(121)[None,:]<=first[:,None]
    pair=(pred[:,1]-pred[:,0])/scale
    losses['physical_pre_influence']=(pair.square().mean(-1)*before).sum()/before.sum().clamp_min(1)
    onset=torch.stack([original['onset'],edited['onset']],1);onset_mask=labels['onset_mask']*mask
    losses['onset_time']=((onset-labels['onset']).square()*onset_mask).sum()/onset_mask.sum().clamp_min(1)
    response=torch.stack([original['response'],edited['response']],1);contact_mask=labels['contacts']*mask
    losses['local_response']=(((response-labels['response'])/torch.tensor([5.,5.,20.])).square().mean(-1)*contact_mask).sum()/contact_mask.sum().clamp_min(1)
    dv=pred[:,:,1:,2:4]-pred[:,:,:-1,2:4]-pred.new_tensor([0,-9.81/60]);absent=labels['contacts'].sum(-1)==0
    losses['unsupported_velocity']=(dv.square().sum(-1)*absent).sum()/absent.sum().clamp_min(1)
    return losses,truth,context,labels,mask


def train(root, protocol, cell, seed):
    config=json.loads(protocol.read_text());out=root/cell/f'seed-{seed}'
    if (out/'model.pt').exists():raise ValueError(f'refusing to overwrite {out}')
    root.mkdir(parents=True,exist_ok=True)
    frozen=root/'training-protocol.json'
    if frozen.exists() and frozen.read_bytes()!=protocol.read_bytes():raise ValueError('protocol changed')
    if not frozen.exists():frozen.write_bytes(protocol.read_bytes())
    manifest,data=load_data(Path(config['data'])/'train');_,val=load_data(Path(config['data'])/'validation')
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);torch.set_num_threads(4);rng=np.random.default_rng(seed)
    model=ArchitectureModel('A');optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
    weights=json.loads(Path('configs/architecture-experiment.json').read_text())['shared_losses']
    sampler=BalancedTransitions(data,seed) if cell=='P4' else None
    n=config['batch_parents'];updates=config['cells'][cell];history=[];started=perf_counter();local_examples=0;extra_edges=0
    for update in range(1,updates+1):
        parents=rng.choice(len(data['states']),n,replace=False);branches=rng.choice([0,1,2,3,5,6,7,8],n)
        original,edited=paired_rollout(model,data['states'][parents,4,0],data['contexts'][parents,4],data['contexts'][parents,branches])
        losses,truth,context,labels,mask=common_losses(original,edited,data,parents,branches)
        if cell=='P4':
            state,c,y=sampler.sample(int(mask.sum()))
            h,_,_=model.representation(state,c);logits=model.phase(h[:,0])
            ce=F.cross_entropy(logits,y,reduction='none')
            losses['phase']=torch.stack([ce[y==k].mean() for k in range(4) if (y==k).any()]).mean()
            extra_edges+=len(y)
        else:
            if cell=='P3':
                state=truth[:,:,:-1].flatten(0,2)
                c=context[:,:,None].expand(-1,-1,120,-1,-1).flatten(0,2)
                h,_,_=model.representation(state,c);logits=model.phase(h).reshape(*mask.shape,4);extra_edges+=mask.numel()
            else:logits=torch.stack([original['phase'],edited['phase']],1)
            losses['phase']=phase_objective(logits,labels['phase'],mask,'count' if cell in ['P0','P1'] else 'weights')
        local_examples+=int(mask.sum())
        # Preserve original addition order so P0 reproduces its optimization prefix.
        loss=sum(weights[k]*losses[k] for k in ['physical','physical_difference','physical_pre_influence','phase','onset_time','local_response','unsupported_velocity'])
        if not torch.isfinite(loss):raise FloatingPointError(f'{cell}/{seed}/{update}')
        gradients={}
        if update in [1,200,updates]:
            for key in ['physical','phase']:
                g=torch.autograd.grad(weights[key]*losses[key],model.representation.edge[0].weight,retain_graph=True)[0]
                gradients[key]=float(g.norm())
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
        if update==1 or update%50==0 or update==updates:
            rmse,_=validation(model,val)
            row=dict(update=update,validation_vector_rmse_m=rmse,losses={k:float(v.detach()) for k,v in losses.items()},gradient_norms=gradients,seconds=perf_counter()-started)
            history.append(row);print(json.dumps(dict(cell=cell,seed=seed,**row)),flush=True)
        if update==updates or (cell=='P1' and update==200):
            payload=dict(model=model.state_dict(),name='Structured architecture experiment',variant='A',cell=cell,seed=seed,epoch=update,dataset_sha256=manifest['sha256'],protocol_sha256=digest(protocol))
            torch.save(payload,out/('model.pt' if update==updates else 'update-200.pt'))
    (out/'training.json').write_text(json.dumps(dict(cell=cell,seed=seed,updates=updates,history=history,seconds=perf_counter()-started,processed_physical_intervals=updates*n*2*120,phase_examples=local_examples,extra_phase_encoded_edges=extra_edges,parameters=sum(p.numel() for p in model.parameters()),protocol_sha256=digest(protocol),dataset_sha256=manifest['sha256']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/contact-learning'));p.add_argument('--protocol',type=Path,default=Path('configs/contact-training.json'));p.add_argument('--cells',nargs='+',default=['P0','P1','P2','P3','P4']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);a=p.parse_args()
    for seed in a.seeds:
        for cell in a.cells:train(a.root,a.protocol,cell,seed)
