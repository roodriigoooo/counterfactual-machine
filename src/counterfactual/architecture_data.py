"""Disjoint parent pools and event/response targets, frozen before fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .geometry import candidates, apply_edit
from .schemas import EditSpec
from .scenes import generate_scene
from .recurrent import pack
from .simulator import simulate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, protocol):
    config = json.loads(protocol.read_text())
    output.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'validation', 'test']:
        folder = output/split
        if (folder/'manifest.json').exists():
            manifest = json.loads((folder/'manifest.json').read_text())
            if manifest['protocol_sha256'] != digest(protocol) or manifest['sha256'] != digest(folder/'paired.npz'):
                raise ValueError('existing data provenance mismatch')
            continue
        folder.mkdir(exist_ok=True)
        start, count = config['parent_pools'][split]
        arrays = {k: [] for k in ['states', 'contacts', 'contexts', 'phase', 'onset', 'onset_mask', 'response']}
        records, audits = [], []
        for seed in range(start, start+count):
            scene = generate_scene(seed)
            grid, valid, _ = candidates(scene, 9)
            edits = np.where(valid, grid, 0.)
            row = {k: [] for k in arrays}
            for b, edit in enumerate(edits):
                tr = simulate(scene, float(edit), record_responses=True)
                c = np.zeros((120, 5), np.float32); c[:, :len(tr.object_ids)] = tr.contacts
                previous = np.vstack([np.zeros((1, 5)), c[:-1]]).astype(bool)
                phase = np.where(c, np.where(previous, 2, 1), np.where(previous, 3, 0))
                onset, onset_mask = np.zeros_like(c), np.zeros_like(c)
                response = np.zeros((120, 5, 3), np.float32)
                for e in tr.events:
                    if e['phase'] != 'begin': continue
                    t = min(119, max(0, int(np.ceil(e['time']*60-1e-7))-1))
                    j = tr.object_ids.index(next(o for o in e['objects'] if o != scene.ball.id))
                    if not onset_mask[t, j]: onset[t, j] = e['time']*60-t
                    onset_mask[t, j] = 1
                for e in tr.responses:
                    t = min(119, max(0, int(np.ceil(e['time']*60-1e-7))-1)); j = tr.object_ids.index(e['object'])
                    n = -np.asarray(e['normal_ball_to_static']); tangent = np.array([-n[1], n[0]])
                    impulse = np.asarray(e['impulse_on_ball'])/scene.ball.mass
                    jn, jt = float(impulse@n), float(impulse@tangent)
                    response[t, j] += [jn, jt, -2*jt/scene.ball.radius]
                edited = apply_edit(scene, EditSpec(object_id=scene.bumper.id, displacement=float(edit)))
                for k, v in dict(states=tr.states, contacts=c, contexts=pack(edited)[1], phase=phase,
                                  onset=onset, onset_mask=onset_mask, response=response).items(): row[k].append(v)
                if seed < start+8 and b in [0, 4, 8]:
                    fine = simulate(scene, float(edit), physics_hz=3840)
                    error = np.linalg.norm(tr.states[:, :2]-fine.states[:, :2], axis=-1)
                    onsets = [e for e in tr.events if e['phase']=='begin']
                    fine_onsets = [e for e in fine.events if e['phase']=='begin']
                    audits.append(dict(parent_id=scene.parent_id, edit_m=float(edit), max_m=float(error.max()),
                                       rms_m=float(np.sqrt(np.mean(error**2))), onsets=len(onsets), fine_onsets=len(fine_onsets),
                                       sensitive=bool(error.max()>.05 or len(onsets)!=len(fine_onsets))))
            for k in arrays: arrays[k].append(row[k])
            records.append(dict(scene=scene.model_dump(), split=split, edits_m=edits.tolist(), requested_valid=valid.tolist()))
            if (seed-start)%16 == 0: print('data', split, seed, flush=True)
        np.savez_compressed(folder/'paired.npz', **{k: np.asarray(v, np.float32) for k, v in arrays.items()})
        (folder/'manifest.json').write_text(json.dumps(dict(split=split, records=records, sha256=digest(folder/'paired.npz'),
                                                          protocol_sha256=digest(protocol), numerical_audit=audits), indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data/architecture'));p.add_argument('--protocol',type=Path,default=Path('configs/architecture-experiment.json'))
    a=p.parse_args();build(a.output,a.protocol)
