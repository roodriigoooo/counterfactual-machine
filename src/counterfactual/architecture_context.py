"""Fixed diagnostic examples: anticipation can change before physical influence."""
import json
from pathlib import Path
import numpy as np
import torch
from .architecture import ArchitecturePredictor
from .contact_suite import fixtures
from .recurrent import pack
from .geometry import apply_edit
from .schemas import EditSpec
from .simulator import simulate


def record(root):
    predictor=ArchitecturePredictor(root/'D/seed-0/model.pt');rows=[]
    for index,label in [(1,'Contact created'),(8,'Irrelevant edit')]:
        scene=fixtures()[index];edit=.12;changed=apply_edit(scene,EditSpec(displacement=edit))
        initial,c0=pack(scene);_,cu=pack(changed)
        with torch.inference_mode():
            a,b=predictor.model.paired(torch.tensor(initial[None]),torch.tensor(c0[None]),torch.tensor(cu[None]))
        truth0,truthu=simulate(scene),simulate(scene,edit)
        contacts=[e['time'] for e in truthu.events if e['phase']=='begin' and scene.bumper.id in e['objects']]
        rows.append(dict(label=label,scene=scene.model_dump(),edit_m=edit,contact_time_s=min(contacts) if contacts else None,
             times_s=truth0.times.tolist(),context_times_s=truth0.times[:-1].tolist(),
             context_difference=(b['hidden']-a['hidden']).flatten(2).norm(dim=-1)[0].tolist(),
             predicted_effect_radii=((b['states']-a['states'])[...,:2].norm(dim=-1)[0]/scene.ball.radius).tolist(),
             actual_effect_radii=(np.linalg.norm(truthu.states[:,:2]-truth0.states[:,:2],axis=-1)/scene.ball.radius).tolist()))
    (root/'context-diagnostics.json').write_text(json.dumps(dict(model=predictor.identity,scope='fixed development fixtures; context norm is internal and not an edit cost or cross-model metric',rows=rows),indent=2))

if __name__=='__main__':record(Path('artifacts/architecture'))
