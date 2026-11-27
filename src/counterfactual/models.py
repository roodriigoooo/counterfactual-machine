"""E0: exploratory, permutation-invariant supervised trajectory emulator.

This fast baseline is deliberately distinct from research variants A/B/C/D.
It pools initial ball/static relations and decodes an entire future in one pass.
There are no simulator calls or true future inputs, and no autoregressive edges.
"""
import numpy as np
import torch
from torch import nn
from .data import features
from .geometry import apply_edit
from .schemas import EditSpec

STATE_SCALE = [3., 3., 5., 5., 20., 20.]


def load_predictor(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["name"] == "Structured architecture experiment":
        from .architecture import ArchitecturePredictor
        return ArchitecturePredictor(checkpoint)
    if payload["name"] == "Contact mechanics experiment":
        from .contact_mechanics import MechanicsPredictor
        return MechanicsPredictor(checkpoint)
    if payload["name"] == "A recurrent interaction prototype":
        from .recurrent import RecurrentPredictor
        return RecurrentPredictor(checkpoint)
    return Predictor(checkpoint)


class TrajectoryBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.relation = nn.Sequential(nn.Linear(20, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU())
        self.encoder = nn.Sequential(nn.Linear(106, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.path = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 120*6))
        self.contact = nn.Sequential(nn.Linear(224, 96), nn.SiLU(), nn.Linear(96, 120))
        nn.init.normal_(self.path[-1].weight, std=.001)
        nn.init.zeros_(self.path[-1].bias)
        self.register_buffer("scale", torch.tensor(STATE_SCALE))

    def forward(self, g, r, mask):
        rel = self.relation(torch.cat([g[:, None, :].expand(-1, r.shape[1], -1), r], dim=-1))
        pooled = (rel*mask[..., None]).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)
        z = self.encoder(torch.cat([g, pooled], dim=-1))
        correction = self.path(z).reshape(-1, 120, 6)*self.scale
        t = torch.arange(1, 121, device=g.device, dtype=g.dtype)[None, :]/60
        x, y = g[:, 0:1]*6, g[:, 1:2]*4
        vx, vy = g[:, 2:3]*5, g[:, 3:4]*5
        angle, spin = g[:, 5:6]*10, g[:, 6:7]*20
        prior = torch.stack([x+vx*t, y+vy*t-9.81*t*(t-1/1920)/2,
                             vx.expand_as(t.expand(g.shape[0], -1)), vy-9.81*t,
                             angle+spin*t, spin.expand(g.shape[0], 120)], dim=-1)
        initial = torch.cat([x, y, vx, vy, angle, spin], dim=1)[:, None, :]
        states = torch.cat([initial, prior+correction], dim=1)
        logits = self.contact(torch.cat([z[:, None, :].expand(-1, r.shape[1], -1), rel], -1)).transpose(1, 2)
        return states, logits


class Predictor:
    def __init__(self, checkpoint):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = TrajectoryBaseline().eval()
        self.model.load_state_dict(payload["model"])
        self.identity = {k: payload[k] for k in ["name", "seed", "epoch", "dataset_sha256"]}
        self.device = "cpu"

    @torch.inference_mode()
    def forecast(self, scene, edits):
        packed = [features(apply_edit(scene, EditSpec(object_id=scene.bumper.id, displacement=float(u)))) for u in edits]
        g, r, m = [torch.tensor(np.stack([p[i] for p in packed])) for i in range(3)]
        states, logits = self.model(g, r, m)
        return states.numpy(), torch.sigmoid(logits).numpy()
