import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .subinterval import SubintervalPredictor
from .schemas import SceneSpec
from .contact_suite import match_events
from .architecture_data import digest


def run():
    torch.set_num_threads(4);root=Path('artifacts/integrator-study');root.mkdir(exist_ok=True)
    protocol=Path('configs/contact-integrator.json');frozen=root/'protocol.json'
    if frozen.exists() and frozen.read_bytes()!=protocol.read_bytes():raise ValueError('protocol changed')
    if not frozen.exists():frozen.write_bytes(protocol.read_bytes())
    fixtures=json.loads(Path('artifacts/contact-suite/suite.json').read_text())['records']
    manifest=json.loads(Path('data/episodes/validation/manifest.json').read_text());data=np.load('data/episodes/validation/paired.npz')
    for substeps in [1,4,16]:
        out=root/f'substeps-{substeps}.json'
        if out.exists():continue
        predictor=SubintervalPredictor(substeps);rows=[]
        cases=[dict(scene=r['scene'],edit_m=r['edit_m'],states=r['states'],onsets=r['onsets'],kind='suite',sensitive=r['numerically_sensitive']) for r in fixtures if r['valid']]
        cases += [dict(scene=r['scene'],edit_m=0.,states=data['states'][i,4],onsets=[dict(time=e['time_s'],objects=['ball',e['object']]) for e in r['events'][4] if e['phase']=='begin'],kind=r['stratum'],sensitive=None) for i,r in enumerate(manifest['records'])]
        for case in cases:
            scene=SceneSpec.model_validate(case['scene']);started=perf_counter();result=predictor.detail(scene,[case['edit_m']]);elapsed=perf_counter()-started
            ids=sorted([scene.bumper.id,*[s.id for s in scene.segments]]);events={}
            for j,obj in enumerate(ids):
                truth=[e['time'] for e in case['onsets'] if obj in e['objects']]
                predictions=[e['time_s'] for e in result['onsets'][0] if e['object_index']==j]
                events[obj]=match_events(truth,predictions)
            error=np.linalg.norm(result['states'][0,:,:2]-np.asarray(case['states'])[:,:2],axis=-1)/scene.ball.radius
            rows.append(dict(parent_id=scene.parent_id,edit_m=case['edit_m'],kind=case['kind'],sensitive=case['sensitive'],position_mean_radii=float(error.mean()),position_p95_radii=float(np.quantile(error,.95)),seconds=elapsed,events=events,**{k:result[k] for k in ['multiple_onset_intervals','multiple_eligible_subintervals','tied_subintervals']}))
        out.write_text(json.dumps(dict(substeps=substeps,protocol_sha256=digest(protocol),rows=rows),indent=2));print('integrator',substeps,flush=True)


if __name__=='__main__':run()
