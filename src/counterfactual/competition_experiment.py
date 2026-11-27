"""Matched E1/E2 begin-event experiment on verified competing surfaces."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from .architecture import conditions, integrate
from .architecture_data import digest
from .competition_scenes import DATA, PROTOCOL, REPORT, begins, disable_surface
from .episode_model import EpisodeModel
from .phase_evaluation import binary
from .recurrent import pack
from .schemas import SceneSpec
from .simulator import simulate


def load_split(folder, protocol_hash):
    manifest = json.loads((folder/"manifest.json").read_text())
    if not manifest["gate_passed"] or manifest["protocol_sha256"] != protocol_hash:
        raise ValueError(f"Exposure gate or split binding failed in {folder}")
    path = folder/"branches.npz"
    if digest(path) != manifest["dataset_sha256"]:
        raise ValueError(f"Branch archive changed in {folder}")
    with np.load(path) as archive:
        data = {k: torch.from_numpy(archive[k]) for k in archive.files}
    return manifest, data


def surface_ids(scene):
    return sorted([scene.bumper.id, *(s.id for s in scene.segments)])


def witness_batch(manifest, data):
    """Two worlds share the entire pre-contact state. Their outcome labels differ."""
    pairs, descriptions, legal = [], [], []
    for parent, record in enumerate(manifest["records"]):
        scene = SceneSpec.model_validate(record["scene"])
        audit = record["audits"]["1920"]
        tick, winner, loser = audit["tick"], audit["winner"], audit["omitted"]
        original = next(b["row"] for b in record["branches"] if b["edit_m"] == 0)
        state = data["states"][original,tick]
        ids = surface_ids(scene)
        isolated = disable_surface(scene, winner)
        counterfactual = simulate(isolated)
        targets = torch.zeros(5)
        for event in begins(counterfactual,tick):
            obj = next(o for o in event["objects"] if o != scene.ball.id)
            targets[surface_ids(isolated).index(obj)] = 1
        if not targets[surface_ids(isolated).index(loser)]:
            raise ValueError("Simulator no longer reproduces a witness")
        intact = (state, data["contexts"][original], data["active_start"][original,tick], data["begin"][original,tick])
        alternate = (state, torch.from_numpy(pack(isolated)[1]), torch.zeros(5), targets)
        pairs.extend([intact,alternate])
        descriptions.append(dict(parent_id=scene.parent_id, winner=winner, winner_index=ids.index(winner),
                                 loser_index=ids.index(loser), recovered_index=surface_ids(isolated).index(loser),
                                 nearest=audit["nearest"], earliest=audit["candidate_ids"][np.argmin(audit["times_s"])]))
        # A stronger in-domain example keeps the ramp present and changes only a
        # legal bumper displacement. Match the whole pre-contact prefix exactly.
        if loser == "ramp":
            for branch in record["branches"]:
                if not branch["valid"] or not branch["edit_m"]:
                    continue
                row = branch["row"]
                if (data["begin"][row,tick,ids.index(loser)] and
                    torch.equal(data["states"][original,:tick+1],data["states"][row,:tick+1]) and
                    not data["begin"][row,:tick].any()):
                    legal.append((parent,row,tick,ids.index(loser),branch["edit_m"]))
                    break  # first valid flip per independent parent
    return tuple(torch.stack(column) for column in zip(*pairs)), descriptions, legal


def ordinary_batch(data, positive, count, rng):
    rows = rng.integers(len(data["states"]),size=count)
    ticks = rng.integers(120,size=count)
    # Begin-centered exposure is training-only. Test all validation intervals.
    idx = rng.integers(len(positive),size=count//2)
    rows[:len(idx)], ticks[:len(idx)] = positive[idx].T
    return data["states"][rows,ticks], data["contexts"][rows], data["active_start"][rows,ticks], data["begin"][rows,ticks]


def logits(model,batch):
    return model.readout(*batch[:3])[0][...,1]


def loss(scores, batch):
    mask = batch[1][...,9].bool()
    return F.binary_cross_entropy_with_logits(scores[mask], batch[3][mask])


def fit(cell, seed, config, witness, ordinary):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EpisodeModel(cell)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    positive = torch.nonzero(ordinary["begin"].any(-1)).numpy()
    n = config["pairs_per_batch"]
    for _ in range(config["updates"]):
        parents = rng.choice(len(witness[0])//2,size=n,replace=False)
        rows = np.column_stack([2*parents,2*parents+1]).ravel()
        paired = tuple(x[rows] for x in witness)
        background = ordinary_batch(ordinary,positive,n,rng)
        batch = tuple(torch.cat([x,y]) for x,y in zip(paired,background))
        scores = logits(model,batch)
        objective = .5*(loss(scores[:len(rows)],paired)+loss(scores[len(rows):],background))
        opt.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
    return model.eval()


@torch.inference_mode()
def local_results(model, batch, descriptions, legal, data):
    scores = logits(model,batch)
    rows = []
    for i,desc in enumerate(descriptions):
        idx = [desc["winner_index"],desc["loser_index"]]
        selected = scores[2*i,idx]
        probs = selected.sigmoid()
        recovered = scores[2*i+1,desc["recovered_index"]].sigmoid().item()
        rows.append(dict(**desc, bce=F.binary_cross_entropy_with_logits(selected,selected.new_tensor([1.,0.])).item(),
                         winner_probability=probs[0].item(),loser_probability=probs[1].item(),
                         recovered_probability=recovered,legal_edit=None))
    for parent,row,tick,index,edit in legal:
        score = logits(model, (data['states'][row:row+1,tick],data['contexts'][row:row+1],
                               data['active_start'][row:row+1,tick]))
        rows[parent]['legal_edit'] = dict(edit_m=edit,ramp_probability=score[0,index].sigmoid().item())
    return rows


@torch.inference_mode()
def ordinary_results(model,data):
    truth = data["begin"].bool()
    valid = data["contexts"][:,None,:,9].expand_as(truth).bool()
    scores = {"exact":[], "release":[]}
    for rows in torch.arange(len(truth)).split(48):
        context = data["contexts"][rows]
        state = data["states"][rows,0]
        active = torch.zeros_like(context[...,9])
        coefficients = torch.stack([context[...,8], torch.ones_like(active)],-1)
        release = []
        for tick in range(120):
            release.append(model.readout(state,context,active)[0][...,1])
            _,geometry = conditions(state,context)
            state,_ = integrate(state,context,geometry,coefficients)
            active = geometry[1].float()
        scores["release"].append(torch.stack(release,1))
        shape = (len(rows),120,5)
        local = logits(model,(data["states"][rows,:-1].flatten(0,1),
                              context[:,None].expand(-1,120,-1,-1).flatten(0,1),
                              data['active_start'][rows].flatten(0,1)))
        scores["exact"].append(local.reshape(shape))
    result = {}
    for name, chunks in scores.items():
        score = torch.cat(chunks)[valid]
        labels = truth[valid]
        result[name] = dict(**binary(labels.numpy(),(score>=0).numpy()),
                            bce=F.binary_cross_entropy_with_logits(score,labels.float()).item())
    return result


def run(data_root=DATA, root=REPORT, protocol=PROTOCOL):
    torch.set_num_threads(2)
    config = json.loads(protocol.read_text())
    protocol_hash = digest(protocol)
    splits = {name:load_split(data_root/name,protocol_hash) for name in config['splits']}
    manifest_hashes = {name:digest(data_root/name/"manifest.json") for name in splits}
    binding = dict(protocol_sha256=protocol_hash,manifest_sha256=manifest_hashes)
    root.mkdir(parents=True,exist_ok=True)
    train,_,_ = witness_batch(*splits["train"])
    evaluation,desc,legal = witness_batch(*splits["validation"])
    for seed in config["comparison"]["seeds"]:
        for cell in ('E1','E2'):
            target = root/cell/f"seed-{seed}"
            if target.exists():
                saved = json.loads((target/"evaluation.json").read_text())
                if saved["binding"] != binding or saved['checkpoint_sha256'] != digest(target/"model.pt"):
                    raise ValueError(f"Run binding changed in {target}")
                continue
            model = fit(cell,seed,config['comparison'],train,splits["ordinary_train"][1])
            metrics = dict(binding=binding,cell=cell,seed=seed,
                           parents=local_results(model,evaluation,desc,legal,splits['validation'][1]),
                           ordinary=ordinary_results(model,splits["ordinary_validation"][1]))
            target.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
                folder=Path(tmp)/'run'
                folder.mkdir()
                torch.save(dict(model=model.state_dict(),seed=seed,cell=cell,**binding),folder/"model.pt")
                metrics["checkpoint_sha256"]=digest(folder/"model.pt")
                (folder/"evaluation.json").write_text(json.dumps(metrics,indent=2))
                folder.rename(target)
            print(cell,seed,float(np.mean([row['bce'] for row in metrics['parents']])),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--output",type=Path,default=REPORT)
    args=parser.parse_args()
    run(args.data,args.output)
