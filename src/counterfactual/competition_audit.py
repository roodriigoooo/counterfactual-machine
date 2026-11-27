"""Validate split identity, replay the legal edit contrast and compare regenerated data."""
import argparse
import json
from itertools import groupby
from pathlib import Path

import numpy as np
import pymunk
import torch
from .architecture_data import digest
from .competition_experiment import load_split
from .competition_scenes import DATA, PROTOCOL, REPORT, begins
from .schemas import SceneSpec
from .simulator import simulate


def compare_manifests(canonical, replica):
    """Only concurrent callbacks on different surfaces may exchange order."""
    reordered = 0
    for left_record, right_record in zip(canonical['records'], replica['records'], strict=True):
        for left, right in zip(left_record['branches'], right_record['branches'], strict=True):
            if left.get('callbacks') == right.get('callbacks'):
                continue
            # Independently group each timestamp. Retain its position in the
            # stream and the complete ordering within each object's history.
            def canonical_order(events):
                return [event for _, group in groupby(events, key=lambda e:e['time_s'])
                        for event in sorted(group, key=lambda e:e['object'])]
            if canonical_order(left['callbacks']) != canonical_order(right['callbacks']):
                raise ValueError('Regenerated callback history changed')
            reordered += 1
            right['callbacks'] = left['callbacks']
    if canonical != replica:
        raise ValueError('Regenerated manifest differs outside simultaneous callback ordering')
    return reordered


def audit(root, output, replica=None):
    config=json.loads(PROTOCOL.read_text())
    binding=digest(PROTOCOL)
    identities=set()
    physical_scenes=set()
    files={}
    simultaneous_reorderings={}
    for split in config["splits"]:
        folder=root/split
        manifest,data=load_split(folder,binding)
        seen_rows=[]
        for record in manifest["records"]:
            scene=SceneSpec.model_validate(record["scene"])
            geometry=scene.model_dump_json(exclude={'id','parent_id','seed','family'})
            if scene.parent_id in identities or geometry in physical_scenes:
                raise ValueError('A parent or its release geometry occurs twice')
            identities.add(scene.parent_id)
            physical_scenes.add(geometry)
            seen_rows.extend(b['row'] for b in record['branches'] if b['valid'])
        if sorted(seen_rows)!=list(range(len(data["states"]))):
            raise ValueError(f'Overlapping or missing branch rows in {split}')
        for name in ['manifest.json','branches.npz']:
            path=folder/name
            files[str(path.relative_to(root))]=digest(path)
            if replica:
                if name == 'manifest.json':
                    simultaneous_reorderings[split]=compare_manifests(manifest,json.loads((replica/split/name).read_text()))
                elif digest(replica/split/name)!=digest(path):
                    raise ValueError(f'Regenerated array archive differs: {split}/{name}')
    validation=json.loads((root/'validation/manifest.json').read_text())
    examples={r['scene']['parent_id']:r for r in validation['records']}
    selected=json.loads((output/'E2/seed-0/evaluation.json').read_text())['parents']
    flips=[]
    for row in selected:
        if row['legal_edit'] is None:
            continue
        record=examples[row['parent_id']]
        scene=SceneSpec.model_validate(record['scene'])
        tick=record['audits']['1920']['tick']
        edit=row['legal_edit']['edit_m']
        checks={}
        for hz in [1920,3840]:
            base,shifted=(simulate(scene,u,physics_hz=hz) for u in [0.,edit])
            prefix=tick*(hz//60)+1
            original_ramp=any('ramp' in e['objects'] for e in begins(base,tick))
            edited_ramp=any('ramp' in e['objects'] for e in begins(shifted,tick))
            checks[str(hz)]=dict(original_ramp=original_ramp,edited_ramp=edited_ramp,
                                 prefix_equal=np.array_equal(base.physics_positions[:prefix],shifted.physics_positions[:prefix]))
        flips.append(dict(parent_id=scene.parent_id,tick=tick,edit_m=edit,checks=checks))
    result=dict(protocol_sha256=binding, independent_parents=len(identities), dataset_hashes=files,
                array_archives_bitwise_reproduced=True if replica else None,
                branches_with_reordered_simultaneous_callbacks=simultaneous_reorderings,
                environment=dict(torch=torch.__version__,pymunk=pymunk.version,numpy=np.__version__),
                legal_parent_replays=flips)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replica',type=Path)
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--output',type=Path,default=REPORT)
    a=p.parse_args()
    result=audit(a.data,a.output,a.replica)
    (a.output/'audit.json').write_text(json.dumps(result,indent=2))
    print('Checked',result['independent_parents'],'parents and',len(result['legal_parent_replays']),'legal edit pairs')
