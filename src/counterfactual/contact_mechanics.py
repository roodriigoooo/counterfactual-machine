"""Explicit geometric contact admission, separate from learned response/confidence.

Eight ballistic chords approximate acceleration during a 60 Hz interval. Each
chord is intersected analytically with circles/capsules. No simulator is called.
The current version resolves the first surface per interval; this limitation is
measured by the contact suite and is not a general rigid-body solver.
"""
import torch
from torch import nn
from collections import OrderedDict
from .recurrent import InteractionRollout, RecurrentPredictor, DT


class MechanicsPredictor(RecurrentPredictor):
    def __init__(self,checkpoint):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model=MechanicsModel(payload['mechanics'],payload['memory']).eval()
        self.model.load_state_dict(payload['model'])
        self.identity={k:payload[k] for k in ['name','seed','epoch','dataset_sha256','mechanics','memory','exposure']}
        self.cache=OrderedDict()


def nearest_normal(p, c):
    a,b=c[...,1:3],c[...,3:5];ab=b-a
    f=((p-a)*ab).sum(-1)/ab.square().sum(-1).clamp_min(1e-12)
    d=p-(a+f.clamp(0,1)[...,None]*ab)
    distance=d.norm(dim=-1).clamp_min(1e-9)
    return d/distance[...,None],distance-c[...,5]-c[...,6]


@torch.no_grad()
def swept_admission(state,c):
    t=torch.linspace(0,DT,9,device=state.device,dtype=state.dtype)
    gravity=state.new_tensor([0.,-9.81])
    path=state[:,None,:2]+state[:,None,2:4]*t[None,:,None]+gravity*(t*(t-1/1920)/2)[None,:,None]
    p=path[:,:-1,None,:];d=path[:,1:,None,:]-p
    a=c[:,None,:,1:3];b=c[:,None,:,3:5];radius=(c[...,5]+c[...,6])[:,None,:]
    roots=[]
    for center in [a,b]:
        rel=p-center;aa=d.square().sum(-1).clamp_min(1e-16)
        bb=(rel*d).sum(-1);cc=rel.square().sum(-1)-radius.square()
        disc=bb.square()-aa*cc
        u=(-bb-disc.clamp_min(0).sqrt())/aa
        roots.append(torch.where((disc>=0)&(u>=0)&(u<=1),u,torch.full_like(u,2)))
    ab=b-a;length=ab.norm(dim=-1).clamp_min(1e-9);tangent=ab/length[...,None]
    normal=torch.stack([-tangent[...,1],tangent[...,0]],-1)
    h=((p-a)*normal).sum(-1);hd=(d*normal).sum(-1)
    for side in [-1,1]:
        u=(side*radius-h)/torch.where(hd.abs()>1e-12,hd,torch.ones_like(hd))
        projection=((p+u[...,None]*d-a)*tangent).sum(-1)
        valid=(hd*side<0)&(u>=0)&(u<=1)&(projection>=0)&(projection<=length)&(length>1e-8)
        roots.append(torch.where(valid,u,torch.full_like(u,2)))
    u=torch.stack(roots).amin(0)
    # Already touching/penetrating: separating motion has no new impact.
    n,gap=nearest_normal(state[:,None,:2],c)
    vn=(state[:,None,2:4]*n).sum(-1)
    gn=(gravity*n).sum(-1)
    touch=(gap<=1e-6)&((vn < -1e-6)|((vn.abs()<1e-6)&(gn<0)))
    seconds=(torch.arange(8,device=state.device)[None,:,None]+u)*(DT/8)
    seconds=torch.where(u<=1,seconds,torch.full_like(seconds,DT+1)).amin(1)
    seconds=torch.where(touch,torch.zeros_like(seconds),seconds)
    eligible=(seconds<=DT)&(c[...,9]>0)
    time=seconds.clamp(0,DT)
    point=state[:,None,:2]+state[:,None,2:4]*time[...,None]+gravity*(time*(time-1/1920)/2)[...,None]
    n,_=nearest_normal(point,c)
    return eligible,time,n,gap


class LocalMemory(nn.Module):
    """Parameter count near the GRU, trained as its own memory-free control."""
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(32,96),nn.SiLU(),nn.Linear(96,32),nn.Tanh())
    def forward(self,x,hidden):
        return self.net(x)


class MechanicsModel(InteractionRollout):
    def __init__(self,mechanics='soft',memory='gru'):
        super().__init__();self.mechanics=mechanics;self.memory_kind=memory
        if memory=='local':self.memory=LocalMemory()
        if mechanics not in ['soft','swept'] or memory not in ['gru','local']:raise ValueError('unknown mechanics/memory')

    def step(self,state,c,encoded,memory):
        if self.mechanics=='soft':return super().step(state,c,encoded,memory)
        eligible,time,n,gap=swept_admission(state,c)
        # Earliest admitted surface only; ties aggregate to preserve object symmetry.
        earliest=torch.where(eligible,time,torch.full_like(time,DT+1)).amin(-1,keepdim=True)
        admitted=eligible & ((time-earliest).abs()<1e-7)
        tangent=torch.stack([-n[...,1],n[...,0]],-1)
        gravity=state.new_tensor([0.,-9.81])
        impact_v=state[:,None,2:4]+gravity*time[...,None]
        vn=(impact_v*n).sum(-1)
        # Resting contacts support the gravity impulse over the remaining interval.
        vn=torch.where((time<1e-7)&(vn.abs()<1e-5),((state[:,None,2:4]+gravity*DT)*n).sum(-1),vn)
        vt=(impact_v*tangent).sum(-1)-state[:,None,5]*c[...,6]
        features=torch.stack([gap/c[...,6].clamp_min(.08),vn/5,vt/5,n[...,0],n[...,1],state[:,None,5].expand_as(gap)/20,impact_v[...,0]/5,impact_v[...,1]/5,(-vn*DT-gap)/c[...,6].clamp_min(.08)],-1).clamp(-50,50)
        edge=self.edge(torch.cat([encoded,features],-1))
        hidden=self.memory(edge.flatten(0,1),memory.flatten(0,1)).reshape_as(memory)
        hidden=hidden*eligible[...,None]
        raw=self.response(hidden)
        # Geometry authorizes mechanics. Learned scores are auxiliary labels,
        # never multipliers for fractional collisions and never a 0.5 force gate.
        normal=(1+raw[...,1].sigmoid())*(-vn).clamp_min(0)*admitted
        jt=-raw[...,2].sigmoid()*vt/3
        bound=c[...,7]*normal
        jt=jt.clamp(-bound,bound)*admitted
        impulse=normal[...,None]*n+jt[...,None]*tangent
        # Only bounded resolution of existing penetration, never a learned shift.
        correction=(-gap-.001).clamp(0,.02)*admitted
        position=state[:,:2]+state[:,2:4]*DT+gravity*(DT*(DT-1/1920)/2)+(impulse*(DT-time)[...,None]+correction[...,None]*n).sum(1)
        velocity=state[:,2:4]+gravity*DT+impulse.sum(1)
        dw=-2*jt/c[...,6].clamp_min(.08)
        spin=state[:,5]+dw.sum(1)
        angle=state[:,4]+state[:,5]*DT+(dw*(DT-time)).sum(1)
        logits=torch.where(eligible,raw[...,0],torch.full_like(gap,-30))
        return torch.cat([position,velocity,angle[:,None],spin[:,None]],-1),hidden,logits
