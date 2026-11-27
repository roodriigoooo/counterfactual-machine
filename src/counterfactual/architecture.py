"""Matched direct/difference predictors with optional contextual joint embeddings.

The physical integrator and admission are identical in all cells and the analytic
control. True future states enter loss construction only, never these rollouts.
"""
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from .contact_mechanics import swept_admission, LocalMemory
from .recurrent import pack, DT
from .geometry import apply_edit
from .schemas import EditSpec


def conditions(state, c):
    eligible, time, normal, gap = swept_admission(state, c)
    earliest = torch.where(eligible, time, torch.full_like(time, DT+1)).amin(-1, keepdim=True)
    admitted = eligible & ((time-earliest).abs() < 1e-7)
    tangent = torch.stack([-normal[..., 1], normal[..., 0]], -1)
    gravity = state.new_tensor([0., -9.81])
    velocity = state[:, None, 2:4] + gravity*time[..., None]
    vn = (velocity*normal).sum(-1)
    vn = torch.where((time < 1e-7) & (vn.abs() < 1e-5),
                     ((state[:, None, 2:4]+gravity*DT)*normal).sum(-1), vn)
    vt = (velocity*tangent).sum(-1)-state[:, None, 5]*c[..., 6]
    features = torch.stack([gap/c[..., 6].clamp_min(.08), vn/5, vt/5,
                            normal[..., 0], normal[..., 1], state[:, None, 5].expand_as(gap)/20,
                            velocity[..., 0]/5, velocity[..., 1]/5,
                            (-vn*DT-gap)/c[..., 6].clamp_min(.08)], -1).clamp(-50, 50)
    return features, (eligible, admitted, time, normal, tangent, gap, vn, vt)


def integrate(state, c, geometry, coefficients):
    _, admitted, time, n, tangent, gap, vn, vt = geometry
    normal = (1+coefficients[..., 0])*(-vn).clamp_min(0)*admitted
    jt = -coefficients[..., 1]*vt/3
    bound = c[..., 7]*normal
    jt = jt.clamp(-bound, bound)*admitted
    impulse = normal[..., None]*n+jt[..., None]*tangent
    correction = (-gap-.001).clamp(0, .02)*admitted
    gravity = state.new_tensor([0., -9.81])
    position = state[:, :2]+state[:, 2:4]*DT+gravity*(DT*(DT-1/1920)/2)
    position = position+(impulse*(DT-time)[..., None]+correction[..., None]*n).sum(1)
    velocity = state[:, 2:4]+gravity*DT+impulse.sum(1)
    dw = -2*jt/c[..., 6].clamp_min(.08)
    spin = state[:, 5]+dw.sum(1)
    angle = state[:, 4]+state[:, 5]*DT+(dw*(DT-time)).sum(1)
    return torch.cat([position, velocity, angle[:, None], spin[:, None]], -1), torch.stack([normal, jt, dw], -1)


class Representation(nn.Module):
    def __init__(self):
        super().__init__()
        self.static = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.edge = nn.Sequential(nn.Linear(41, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.local = LocalMemory()
        self.dynamic = nn.Sequential(nn.Linear(41, 32), nn.SiLU(), nn.Linear(32, 16))

    def encode_static(self, c):
        return self.static(torch.stack([c[..., 0], c[..., 5], c[..., 6], c[..., 7], c[..., 8]], -1))

    def forward(self, state, c, encoded=None):
        features, geometry = conditions(state, c)
        encoded = self.encode_static(c) if encoded is None else encoded
        edge = self.edge(torch.cat([encoded, features], -1))
        h = self.local(edge, None)*c[..., 9:10]
        # Keep anticipatory context even when no physical interaction is admitted.
        z = self.dynamic(torch.cat([h, features], -1))*c[..., 9:10]
        return h, z, geometry


class ArchitectureModel(nn.Module):
    def __init__(self, variant='A'):
        super().__init__()
        if variant not in 'ABCD' or len(variant) != 1:
            raise ValueError('expected A/B/C/D')
        self.variant = variant
        self.representation = Representation()
        self.absolute = nn.Linear(32, 2)
        self.contextual = nn.Sequential(nn.Linear(96, 32), nn.SiLU(), nn.Linear(32, 2))
        self.phase = nn.Linear(32, 4)  # absent, onset, persistence, separation
        self.onset_time = nn.Linear(32, 1)
        self.future = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 16))

    def response(self, h, reference=None):
        if reference is None:
            reference = h
        # For B/D this equals R(h0,h0) + [G(hu,h0,hu-h0)-G(h0,h0,0)].
        # The original branch uses exactly the same R as A/C. Zero edit is identity.
        return self.absolute(reference)+self.contextual(torch.cat([h, reference, h-reference], -1))

    def rollout(self, initial, c, steps=120, reference=None, auxiliary=False):
        encoded = self.representation.encode_static(c)
        states, hs, scores, phases, times, responses, futures = [initial], [], [], [], [], [], []
        delta = torch.zeros_like(initial)
        for t in range(steps):
            state = reference['states'][:, t]+delta if reference is not None else states[-1]
            h, _, geo = self.representation(state, c, encoded)
            h0 = reference['hidden'][:, t] if reference is not None else None
            raw = self.response(h, h0)
            following, response = integrate(state, c, geo, raw.sigmoid())
            if reference is not None:
                delta = following-reference['states'][:, t+1]
                following = reference['states'][:, t+1]+delta
            logits = self.phase(h)
            score = logits.softmax(-1)[..., 1:3].sum(-1)*geo[0]
            states.append(following); hs.append(h); scores.append(score)
            if auxiliary:
                phases.append(logits); times.append(self.onset_time(h).sigmoid().squeeze(-1))
                responses.append(response); futures.append(self.future(h))
        result = dict(states=torch.stack(states, 1), hidden=torch.stack(hs, 1), scores=torch.stack(scores, 1))
        if auxiliary:
            result.update(phase=torch.stack(phases, 1), onset=torch.stack(times, 1),
                          response=torch.stack(responses, 1), future=torch.stack(futures, 1))
        return result

    def paired(self, initial, original_context, edited_context, steps=120, auxiliary=False):
        original = self.rollout(initial, original_context, steps, auxiliary=auxiliary)
        edited = self.rollout(initial, edited_context, steps,
                              reference=original if self.variant in 'BD' else None, auxiliary=auxiliary)
        return original, edited


class AnalyticPredictor:
    identity = dict(name='Analytic swept-response control', trainable_parameters=0)

    def __init__(self):
        self.cache = OrderedDict()

    @torch.inference_mode()
    def forecast(self, scene, edits):
        key = (scene.content_hash(), tuple(float(u) for u in edits))
        if key not in self.cache:
            pairs = [pack(apply_edit(scene, EditSpec(object_id=scene.bumper.id, displacement=float(u)))) for u in edits]
            state, c = [torch.tensor(np.stack([p[i] for p in pairs])) for i in [0, 1]]
            states, scores = [state], []
            coefficient = torch.stack([c[..., 8], torch.ones_like(c[..., 8])], -1)
            for _ in range(120):
                _, geo = conditions(states[-1], c)
                following, _ = integrate(states[-1], c, geo, coefficient)
                states.append(following); scores.append(geo[1].float())
            self.cache[key] = (torch.stack(states, 1).numpy(), torch.stack(scores, 1).numpy())
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
        return tuple(v.copy() for v in self.cache[key])


class ArchitecturePredictor:
    def __init__(self, checkpoint):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model = ArchitectureModel(payload['variant']).eval()
        self.model.load_state_dict(payload['model'])
        self.identity = {k: payload[k] for k in ['name', 'variant', 'seed', 'epoch', 'dataset_sha256']}
        self.cache, self.originals = OrderedDict(), OrderedDict()

    @torch.inference_mode()
    def forecast(self, scene, edits):
        key = (scene.content_hash(), tuple(float(u) for u in edits))
        if key not in self.cache:
            pairs = [pack(apply_edit(scene, EditSpec(object_id=scene.bumper.id, displacement=float(u)))) for u in edits]
            initial, c = [torch.tensor(np.stack([p[i] for p in pairs])) for i in [0, 1]]
            reference = None
            if self.model.variant in 'BD':
                skey = scene.content_hash()
                if skey not in self.originals:
                    s0, c0 = pack(scene)
                    self.originals[skey] = self.model.rollout(torch.tensor(s0[None]), torch.tensor(c0[None]))
                    while len(self.originals) > 4:
                        self.originals.popitem(last=False)
                reference = {k: v.expand(len(edits), *v.shape[1:]) for k, v in self.originals[skey].items()}
            result = self.model.rollout(initial, c, reference=reference)
            if reference is not None:
                # Reuse the canonical original exactly, including across batch sizes.
                # This is a known zero edit, not future simulator information.
                zero = torch.tensor([float(u) == 0. for u in edits])
                result['states'][zero] = reference['states'][zero]
                result['scores'][zero] = reference['scores'][zero]
            self.cache[key] = (result['states'].numpy(), result['scores'].numpy())
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
        return tuple(v.copy() for v in self.cache[key])
