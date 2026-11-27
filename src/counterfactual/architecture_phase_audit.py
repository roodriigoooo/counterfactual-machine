"""Post-evaluation diagnosis, with frozen weights and no threshold fitting.

Use original validation trajectories only. Exact-state inputs distinguish a weak
phase readout from accumulated rollout error. This is diagnostic, not a new
architecture comparison or a replacement for the frozen free-running metrics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .architecture import ArchitecturePredictor, conditions
from .architecture_data import digest
from .architecture_training import load_data
from .mechanics_evaluation import classify


@torch.inference_mode()
def audit(data, root):
    torch.set_num_threads(4)
    manifest, arrays = load_data(data)
    state = arrays['states'][:, 4, :-1].flatten(0, 1)
    c = arrays['contexts'][:, 4, None].expand(-1, 120, -1, -1).flatten(0, 1)
    labels = arrays['phase'][:, 4].flatten(0, 1).long()
    truth = arrays['contacts'][:, 4].flatten(0, 1).bool()
    valid = c[..., 9].bool()
    _, geometry = conditions(state, c)
    bins = np.linspace(0, 1, 21)
    result = dict(
        role='Post-evaluation diagnosis; frozen weights; no fitting or threshold change',
        cohort='48 validation parents, original branch, exact simulator states',
        dataset_sha256=manifest['sha256'], threshold=.5,
        histogram_edges=bins.tolist(),
        phase_labels=['absent', 'onset', 'persistence', 'separation'],
        geometric_admission=classify(truth[valid].numpy(), geometry[1][valid].numpy()),
        models={},
    )
    for checkpoint in sorted(root.glob('*/seed-*/model.pt')):
        model = ArchitecturePredictor(checkpoint).model
        h, _, _ = model.representation(state, c)
        probability = model.phase(h).softmax(-1)
        score = probability[..., 1:3].sum(-1) * geometry[0]
        confusion = torch.bincount(
            (4 * labels + probability.argmax(-1))[valid], minlength=16
        ).reshape(4, 4)
        strata = {}
        for label, mask in [('contact', truth & valid), ('no_contact', ~truth & valid)]:
            values = score[mask].numpy()
            strata[label] = dict(
                n=len(values), mean=float(values.mean()), max=float(values.max()),
                histogram=np.histogram(values, bins)[0].tolist(),
            )
        result['models'][str(checkpoint.parent.relative_to(root))] = dict(
            checkpoint_sha256=digest(checkpoint),
            phase_confusion=confusion.tolist(),
            contact=classify(truth[valid].numpy(), (score[valid] >= .5).numpy()),
            scores=strata,
        )
    (root/'phase-audit.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/architecture/validation'))
    parser.add_argument('--root', type=Path, default=Path('artifacts/architecture'))
    args = parser.parse_args()
    audit(args.data, args.root)
