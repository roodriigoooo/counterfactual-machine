"""Audit actual competing-contact exposure, independently of stratum names."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .architecture import conditions
from .architecture_data import digest


@torch.inference_mode()
def audit(data_root,output):
    torch.set_num_threads(4);result={}
    for split in ['train','validation']:
        folder=data_root/split;manifest=json.loads((folder/'manifest.json').read_text())
        with np.load(folder/'paired.npz') as a:
            states=torch.tensor(a['states'][:,:,:-1]);c=torch.tensor(a['contexts'])
        counts=[]
        for state,context,record in zip(states,c,manifest['records']):
            _,geo=conditions(state.flatten(0,1),context[:,None].expand(-1,120,-1,-1).flatten(0,1))
            counts.append(dict(parent_id=record['scene']['parent_id'],stratum=record['stratum'],multiple_candidate_intervals=int((geo[0].sum(-1)>1).sum())))
        result[split]=dict(dataset_sha256=digest(folder/'paired.npz'),multiple_candidate_intervals=sum(r['multiple_candidate_intervals'] for r in counts),parents_with_multiple_candidates=sum(r['multiple_candidate_intervals']>0 for r in counts),parents=counts)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{x:v[x] for x in ['multiple_candidate_intervals','parents_with_multiple_candidates']} for k,v in result.items()}));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/episodes'));p.add_argument('--output',type=Path,default=Path('artifacts/episodes/competition-coverage.json'));a=p.parse_args();audit(a.data,a.output)
