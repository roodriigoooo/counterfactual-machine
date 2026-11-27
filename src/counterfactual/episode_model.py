"""Interpretable contact history and competition; confidence never drives forces."""
import torch
from torch import nn
from .architecture import Representation,conditions,integrate
from .contact_mechanics import nearest_normal
from .recurrent import DT


@torch.no_grad()
def event_geometry(state,c):
    _,geo=conditions(state,c);eligible,_,time,*_=geo
    samples=torch.linspace(0,DT,9,device=state.device,dtype=state.dtype)
    gravity=state.new_tensor([0.,-9.81])
    path=state[:,None,:2]+state[:,None,2:4]*samples[None,:,None]+gravity*(samples*(samples-1/1920)/2)[None,:,None]
    _,gap=nearest_normal(path[:,:,None,:],c[:,None])
    minimum=gap.amin(1)/c[...,6].clamp_min(.08)
    own=torch.stack([eligible.float(),torch.where(eligible,time/DT,0),minimum.clamp(-5,5)],-1)
    times=torch.where(eligible,time,torch.full_like(time,DT+1))
    other=times[:,None,:].expand(-1,c.shape[1],-1).clone()
    eye=torch.eye(c.shape[1],dtype=torch.bool,device=c.device)[None]
    other.masked_fill_(eye,DT+1);earliest_other=other.amin(-1);exists=earliest_other<=DT
    top=times.sort(-1).values;second=top[:,1] if c.shape[1]>1 else torch.full_like(top[:,0],DT+1)
    pair=second<=DT;difference=torch.where(pair,(second-top[:,0])/DT,0)[:,None].expand_as(time)
    competing=torch.stack([exists.float(),torch.where(exists,earliest_other/DT,0),pair[:,None].expand_as(time).float(),difference],-1)
    return own,competing,geo


class EpisodeModel(nn.Module):
    def __init__(self,cell='E0'):
        super().__init__()
        if cell not in ['E0','E1','E2']:raise ValueError('unknown episode cell')
        self.cell=cell;self.representation=Representation()
        self.events=nn.Sequential(nn.Linear(40,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,8))

    def readout(self,state,c,active,encoded=None):
        h,_,_=self.representation(state,c,encoded);own,other,_=event_geometry(state,c)
        episode=active if self.cell!='E0' else torch.zeros_like(active)
        other=other if self.cell=='E2' else torch.zeros_like(other)
        output=self.events(torch.cat([h,own,episode[...,None],other],-1))
        return output[...,:6],output[...,6:].sigmoid()

    def rollout_readout(self,states,c):
        encoded=self.representation.encode_static(c);active=states.new_zeros(c.shape[:2]);logits=[];times=[];history=[]
        for t in range(states.shape[1]-1):
            history.append(active)
            logit,timing=self.readout(states[:,t],c,active,encoded);logits.append(logit);times.append(timing)
            _,next_gap=nearest_normal(states[:,t+1,None,:2],c)
            active=((logit[...,3].sigmoid()>=.5)&(next_gap<=.02)&(c[...,9]>0)).float()
        return torch.stack(logits,1),torch.stack(times,1),torch.stack(history,1)


@torch.inference_mode()
def analytic_paths(initial,c):
    states=[initial];coefficient=torch.stack([c[...,8],torch.ones_like(c[...,8])],-1)
    for _ in range(120):
        _,geo=conditions(states[-1],c);following,_=integrate(states[-1],c,geo,coefficient);states.append(following)
    return torch.stack(states,1)
