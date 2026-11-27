"""Controlled multiple-contact resolution by bounded subinterval subdivision.

The response law is unchanged. Geometry is recomputed after each subinterval;
several sequential responses may occur within one 60 Hz observation interval.
"""
from collections import OrderedDict
import numpy as np
import torch
from .contact_mechanics import nearest_normal
from .architecture import AnalyticPredictor
from .recurrent import DT,pack
from .geometry import apply_edit
from .schemas import EditSpec


@torch.no_grad()
def admission(state,c,dt,physics_dt=1/1920):
    vector = isinstance(dt, torch.Tensor) and dt.ndim > 0
    if vector:
        t = dt[:, None] * torch.linspace(0, 1, 9, device=state.device, dtype=state.dtype)
    else:
        t = torch.linspace(0, dt, 9, device=state.device, dtype=state.dtype)[None]
    gravity=state.new_tensor([0.,-9.81])
    path=state[:,None,:2]+state[:,None,2:4]*t[...,None]+gravity*(t*(t-physics_dt)/2)[...,None]
    p=path[:,:-1,None,:];d=path[:,1:,None,:]-p
    a=c[:,None,:,1:3];b=c[:,None,:,3:5];radius=(c[...,5]+c[...,6])[:,None,:];roots=[]
    for center in [a,b]:
        rel=p-center;aa=d.square().sum(-1).clamp_min(1e-16);bb=(rel*d).sum(-1);cc=rel.square().sum(-1)-radius.square();disc=bb.square()-aa*cc
        u=(-bb-disc.clamp_min(0).sqrt())/aa;roots.append(torch.where((disc>=0)&(u>=0)&(u<=1),u,torch.full_like(u,2)))
    ab=b-a;length=ab.norm(dim=-1).clamp_min(1e-9);tangent=ab/length[...,None];normal=torch.stack([-tangent[...,1],tangent[...,0]],-1)
    h=((p-a)*normal).sum(-1);hd=(d*normal).sum(-1)
    for side in [-1,1]:
        u=(side*radius-h)/torch.where(hd.abs()>1e-12,hd,torch.ones_like(hd));projection=((p+u[...,None]*d-a)*tangent).sum(-1)
        valid=(hd*side<0)&(u>=0)&(u<=1)&(projection>=0)&(projection<=length)&(length>1e-8)
        roots.append(torch.where(valid,u,torch.full_like(u,2)))
    u=torch.stack(roots).amin(0);n,gap=nearest_normal(state[:,None,:2],c);vn=(state[:,None,2:4]*n).sum(-1);gn=(gravity*n).sum(-1)
    touch=(gap<=1e-6)&((vn < -1e-6)|((vn.abs()<1e-6)&(gn<0)))
    seconds=(torch.arange(8,device=state.device)[None,:,None]+u)*((dt[:,None,None] if vector else dt)/8)
    seconds=torch.where(u<=1,seconds,torch.ones_like(seconds)*((dt[:,None,None] if vector else dt)+1)).amin(1);seconds=torch.where(touch,torch.zeros_like(seconds),seconds)
    limit = dt[:,None] if vector else dt
    eligible=(seconds<=limit)&(c[...,9]>0);time=seconds.clamp_min(0).clamp_max(limit)
    point=state[:,None,:2]+state[:,None,2:4]*time[...,None]+gravity*(time*(time-physics_dt)/2)[...,None];n,_=nearest_normal(point,c)
    return eligible,time,n,gap


def contact_response(state, c, time, n, gap, admitted, duration):
    tangent=torch.stack([-n[...,1],n[...,0]],-1);gravity=state.new_tensor([0.,-9.81])
    impact=state[:,None,2:4]+gravity*time[...,None];vn=(impact*n).sum(-1)
    vn=torch.where((time<1e-7)&(vn.abs()<1e-5),((state[:,None,2:4]+gravity*duration[...,None])*n).sum(-1),vn)
    vt=(impact*tangent).sum(-1)-state[:,None,5]*c[...,6]
    normal=(1+c[...,8])*(-vn).clamp_min(0)*admitted;jt=(-vt/3).clamp(-c[...,7]*normal,c[...,7]*normal)*admitted
    impulse=normal[...,None]*n+jt[...,None]*tangent;correction=(-gap-.001).clamp(0,.02)*admitted
    return impulse, correction, -2*jt/c[...,6].clamp_min(.08)


@torch.no_grad()
def step(state,c,dt):
    eligible,time,n,gap=admission(state,c,dt);earliest=torch.where(eligible,time,torch.full_like(time,dt+1)).amin(-1,keepdim=True)
    admitted=eligible&((time-earliest).abs()<1e-7)
    impulse,correction,dw=contact_response(state,c,time,n,gap,admitted,state.new_tensor(dt))
    gravity=state.new_tensor([0.,-9.81])
    position=state[:,:2]+state[:,2:4]*dt+gravity*(dt*(dt-1/1920)/2)+(impulse*(dt-time)[...,None]+correction[...,None]*n).sum(1)
    velocity=state[:,2:4]+gravity*dt+impulse.sum(1)
    following=torch.cat([position,velocity,(state[:,4]+state[:,5]*dt+(dw*(dt-time)).sum(1))[:,None],(state[:,5]+dw.sum(1))[:,None]],-1)
    return following,admitted,time,eligible


class SubintervalPredictor:
    def __init__(self,substeps=4):
        if substeps not in [1,4,16]:raise ValueError('substeps must follow the study protocol')
        self.substeps=substeps;self.cache=OrderedDict();self.identity=dict(name='Analytic subinterval control',substeps=substeps)

    @torch.inference_mode()
    def detail(self,scene,edits):
        key=(scene.content_hash(),tuple(float(e) for e in edits))
        if key not in self.cache:
            pairs=[pack(apply_edit(scene,EditSpec(displacement=float(u)))) for u in edits]
            state,c=[torch.tensor(np.stack([p[i] for p in pairs])) for i in [0,1]];states=[state];scores=[];onsets=[[] for _ in edits]
            previous=torch.zeros(c.shape[:2],dtype=torch.bool);competition=0;ties=0;multiple=0;dt=DT/self.substeps
            for outer in range(120):
                occupied=torch.zeros_like(previous);counts=torch.zeros_like(c[...,0],dtype=torch.int32)
                for inner in range(self.substeps):
                    state,admitted,time,eligible=step(state,c,dt);began=admitted&~previous
                    for batch,edge in torch.nonzero(began).tolist():onsets[batch].append(dict(time_s=outer*DT+inner*dt+float(time[batch,edge]),object_index=edge))
                    counts+=began;occupied|=admitted;previous=admitted
                    competition+=int((eligible.sum(-1)>1).sum());ties+=int((admitted.sum(-1)>1).sum())
                states.append(state);scores.append(occupied.float());multiple+=int((counts.sum(-1)>1).sum())
            self.cache[key]=dict(states=torch.stack(states,1).numpy(),scores=torch.stack(scores,1).numpy(),onsets=onsets,multiple_onset_intervals=multiple,multiple_eligible_subintervals=competition,tied_subintervals=ties)
            while len(self.cache)>4:self.cache.popitem(last=False)
        return self.cache[key]

    def forecast(self,scene,edits):
        result=self.detail(scene,edits);return result['states'].copy(),result['scores'].copy()
