"""Research A prototype: cached context, recurrent local interactions, physical state.

All interaction geometry comes from predicted state. The conservative swept
reachability gate is a shared physical prior, not a true future contact label.
"""
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from .geometry import apply_edit
from .schemas import EditSpec

DT = 1/60


def pack(scene):
    b = scene.ball
    objects = sorted([scene.bumper, *scene.segments], key=lambda o: o.id)
    context = np.zeros((5, 10), np.float32)
    for i, obj in enumerate(objects):
        circle = obj.id == scene.bumper.id
        a, end = (obj.position, obj.position) if circle else (obj.a, obj.b)
        context[i] = [float(circle), *a, *end, obj.radius, b.radius,
                      obj.material.friction*b.material.friction,
                      obj.material.restitution*b.material.restitution, 1]
    state = np.array([*b.position, *b.velocity, b.angle, b.angular_velocity], np.float32)
    return state, context


class InteractionRollout(nn.Module):
    def __init__(self):
        super().__init__()
        self.static = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.edge = nn.Sequential(nn.Linear(41, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.memory = nn.GRUCell(32, 32)
        self.response = nn.Linear(32, 5)  # contact, normal/tangent response, timing, position correction

    def encode(self, c):
        # Translation-independent shape/material cache; prescribed poses remain explicit.
        return self.static(torch.stack([c[..., 0], c[..., 5], c[..., 6], c[..., 7], c[..., 8]], -1))

    def step(self, state, c, encoded, memory):
        p, v = state[:, None, :2], state[:, None, 2:4]
        a, b = c[..., 1:3], c[..., 3:5]
        ab = b-a
        alpha = ((p-a)*ab).sum(-1)/ab.square().sum(-1).clamp_min(1e-8)
        nearest = a+alpha.clamp(0, 1)[..., None]*ab
        relative = p-nearest
        distance = relative.norm(dim=-1).clamp_min(1e-7)
        n = relative/distance[..., None]
        tangent = torch.stack([-n[..., 1], n[..., 0]], -1)
        gap = distance-c[..., 5]-c[..., 6]
        vn = (v*n).sum(-1)
        vt = (v*tangent).sum(-1)-state[:, None, 5]*c[..., 6]
        # Includes acceleration and motion in every direction; retains all pairs.
        reachable = (gap <= v.norm(dim=-1)*DT+9.81*DT*DT+.002) & (c[..., 9] > 0)
        features = torch.stack([gap/c[..., 6].clamp_min(.08), vn/5, vt/5,
                                n[..., 0], n[..., 1], state[:, None, 5].expand_as(gap)/20,
                                v[..., 0].expand_as(gap)/5, v[..., 1].expand_as(gap)/5,
                                (-vn*DT-gap)/c[..., 6].clamp_min(.08)], -1).clamp(-50, 50)
        edge = self.edge(torch.cat([encoded, features], -1))
        hidden = self.memory(edge.flatten(0, 1), memory.flatten(0, 1)).reshape_as(memory)
        hidden = hidden*reachable[..., None]
        raw = self.response(hidden)
        logits = torch.where(reachable, raw[..., 0], torch.full_like(gap, -30))
        probability = logits.sigmoid()*reachable
        # Learned impulse magnitude and friction response, rather than an analytic bounce.
        normal = torch.nn.functional.softplus(raw[..., 1])*((-vn).clamp_min(0)+.15)*probability
        tangential = raw[..., 2].tanh()*(vt.abs()+.15)*probability
        impulse = normal[..., None]*n+tangential[..., None]*tangent
        timing = raw[..., 3].sigmoid()
        correction = raw[..., 4].tanh()*.02*probability
        delta_p = (impulse*(1-timing[..., None])*DT+correction[..., None]*n).sum(1)
        delta_v = impulse.sum(1)
        delta_spin = (-2*tangential/c[..., 6].clamp_min(.08)).sum(1)
        gravity = state.new_tensor([0., -9.81])
        position = state[:, :2]+state[:, 2:4]*DT+gravity*(DT*(DT-1/1920)/2)+delta_p
        velocity = state[:, 2:4]+gravity*DT+delta_v
        spin = state[:, 5]+delta_spin
        angle = state[:, 4]+state[:, 5]*DT+delta_spin*DT/2
        return torch.cat([position, velocity, angle[:, None], spin[:, None]], -1), hidden, logits

    def forward(self, initial, context, steps=120):
        encoded = self.encode(context)
        memory = initial.new_zeros((len(initial), context.shape[1], 32))
        states, contacts = [initial], []
        for _ in range(steps):
            state, memory, logits = self.step(states[-1], context, encoded, memory)
            states.append(state); contacts.append(logits)
        return torch.stack(states, 1), torch.stack(contacts, 1)


class RecurrentPredictor:
    def __init__(self, checkpoint):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model = InteractionRollout().eval()
        self.model.load_state_dict(payload['model'])
        self.identity = {k: payload[k] for k in ['name', 'seed', 'epoch', 'dataset_sha256']}
        self.cache = OrderedDict()

    @torch.inference_mode()
    def forecast(self, scene, edits):
        # Goal-independent cache, bounded by scene/grid requests. No oracle access.
        key = (scene.content_hash(), tuple(float(u) for u in edits))
        if key not in self.cache:
            pairs = [pack(apply_edit(scene, EditSpec(object_id=scene.bumper.id, displacement=float(u)))) for u in edits]
            initial, context = [torch.tensor(np.stack([p[i] for p in pairs])) for i in [0, 1]]
            state, contact = self.model(initial, context)
            self.cache[key] = (state.numpy(), contact.sigmoid().numpy())
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
        return tuple(v.copy() for v in self.cache[key])
