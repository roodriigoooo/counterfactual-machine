import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from .models import TrajectoryBaseline


def train(data: Path, output: Path, epochs=200, seed=0, device="cpu"):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dataset = np.load(data/"paired.npz")
    manifest = json.loads((data/"manifest.json").read_text())
    tensors = {k: torch.tensor(dataset[k], device=device) for k in ["global_features", "relations", "mask", "states", "contacts"]}
    train_ids, val_ids = [np.flatnonzero(dataset["split"] == i) for i in [0, 1]]
    model = TrajectoryBaseline().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    output.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    started = perf_counter()

    def loss_for(ids):
        g, r, m, truth, contacts = [tensors[k][ids].flatten(0, 1) for k in tensors]
        states, logits = model(g, r, m)
        error = (states[:, 1:]-truth[:, 1:])/model.scale
        weights = torch.tensor([1, 1, .2, .2, .02, .02], device=device)
        physical = (error.square()*weights).mean()
        paired = error.reshape(len(ids), 9, 120, 6)
        difference = ((paired-paired[:, 4:5]).square()*weights).mean()
        contact = F.binary_cross_entropy_with_logits(logits, contacts, reduction="none", pos_weight=torch.tensor(5., device=device))
        contact = (contact*m[:, None, :]).sum()/(m.sum()*120)
        loss = physical+.5*difference+.02*contact
        position = (states[:, :, :2]-truth[:, :, :2]).square().sum(-1).mean().sqrt()
        return loss, position

    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for start in range(0, len(train_ids), 16):
            if start == 0:
                order = rng.permutation(train_ids)
            ids = order[start:start+16]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for(ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.inference_mode():
                val, position = loss_for(val_ids)
            row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": float(val),
                   "validation_position_rmse_m": float(position), "seconds": perf_counter()-started}
            history.append(row)
            if float(val) < best:
                best = float(val)
                torch.save({"model": model.cpu().state_dict(), "name": "E0 supervised trajectory baseline",
                            "seed": seed, "epoch": epoch, "dataset_sha256": manifest["sha256"]}, output/"baseline.pt")
                model.to(device)
            if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
                print(json.dumps(row), flush=True)
    summary = {"model": "E0", "architecture": "initial-relation pooling + full-horizon decoder; not research A",
               "parameters": sum(p.numel() for p in model.parameters()), "seed": seed, "epochs": epochs,
               "device": device, "torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
               "dataset_sha256": manifest["sha256"], "best_validation_loss": best,
               "seconds": perf_counter()-started, "history": history}
    (output/"training.json").write_text(json.dumps(summary, indent=2))
    return {k: v for k, v in summary.items() if k != "history"}
