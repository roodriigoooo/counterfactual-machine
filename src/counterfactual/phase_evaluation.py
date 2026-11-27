"""Natural-prevalence event metrics, with raw and geometric controls visible."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .architecture import ArchitecturePredictor, conditions
from .architecture_data import digest
from .architecture_training import load_data
from .mechanics_evaluation import measure


def binary(truth, prediction):
    truth=np.asarray(truth,bool);prediction=np.asarray(prediction,bool)
    tp=int((truth&prediction).sum());fp=int((~truth&prediction).sum());fn=int((truth&~prediction).sum())
    return dict(tp=tp,fp=fp,fn=fn,support=int(truth.sum()),total=truth.size,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def phase_metrics(probability, phase, mask, eligible, admitted, parent_ids):
    prob=probability[mask];labels=phase[mask];parent=parent_ids[mask]
    prediction=prob.argmax(-1);confusion=np.bincount(4*labels+prediction,minlength=16).reshape(4,4)
    support=[dict(intervals=int((labels==k).sum()),parents=int(len(np.unique(parent[labels==k])))) for k in range(4)]
    contact=(labels==1)|(labels==2);score=prob[:,1:3].sum(-1)
    return dict(confusion=confusion.tolist(),phase_support=support,
                phase_f1=[binary(labels==k,prediction==k)['f1'] for k in range(4)],
                raw_contact=binary(contact,score>=.5),
                masked_contact=binary(contact,(score>=.5)&eligible[mask]),
                always_positive_masked=binary(contact,eligible[mask]),
                geometry=binary(contact,admitted[mask]))


@torch.inference_mode()
def evaluate(checkpoint, data_folder):
    torch.set_num_threads(4);manifest,data=load_data(data_folder)
    predictor=ArchitecturePredictor(checkpoint);model=predictor.model
    all_states=[];all_logits=[];all_times=[]
    for idx in torch.arange(len(data['states'])).split(8):
        output=model.rollout(data['states'][idx,:,0].flatten(0,1),data['contexts'][idx].flatten(0,1),auxiliary=True)
        all_states.append(output['states'].reshape(len(idx),9,121,6))
        all_logits.append(output['phase'].reshape(len(idx),9,120,5,4))
        all_times.append(output['onset'].reshape(len(idx),9,120,5))
    predicted=torch.cat(all_states);free_logits=torch.cat(all_logits);free_onset=torch.cat(all_times)
    shape=data['phase'].shape;mask=data['contexts'][:,:,None,:,9].expand(*shape).bool().numpy();truth=data['phase'].long().numpy()
    parents=np.broadcast_to(np.arange(len(truth))[:,None,None,None],shape)
    contexts=data['contexts'][:,:,None].expand(-1,-1,120,-1,-1).flatten(0,2)
    record=dict(checkpoint_sha256=digest(checkpoint),dataset_sha256=manifest['sha256'],threshold=.5,
                label_semantics='Legacy occupancy transitions, not actual begin/separate callback events',metrics={})
    cached={}
    for mode,states in [('exact',data['states']),('release',predicted)]:
        initial=states[:,:,:-1].flatten(0,2);_,geo=conditions(initial,contexts)
        if mode=='exact':
            h,_,_=model.representation(initial,contexts);logits=model.phase(h).reshape(*shape,4);onset=model.onset_time(h).sigmoid().reshape(*shape)
        else:logits=free_logits;onset=free_onset
        prob=logits.softmax(-1).numpy();eligible=geo[0].reshape(*shape).numpy();admitted=geo[1].reshape(*shape).numpy()
        record['metrics'][mode]={}
        for cohort,branch in [('all_branches',slice(None)),('originals',slice(4,5))]:
            record['metrics'][mode][cohort]=phase_metrics(prob[:,branch],truth[:,branch],mask[:,branch],eligible[:,branch],admitted[:,branch],parents[:,branch])
        onset_mask=data['onset_mask'].bool()&torch.from_numpy(mask)
        timing=(onset-data['onset']).abs()[onset_mask].numpy()/60
        record['metrics'][mode]['conditional_onset_time']=dict(n=len(timing),mae_ms=float(timing.mean()*1000),p95_ms=float(np.quantile(timing,.95)*1000))
        if mode=='release':
            scores=prob[...,1:3].sum(-1)*eligible
            for i,row in enumerate(manifest['records']):cached[row['scene']['parent_id']]=(predicted[i].numpy(),scores[i])
    class Cached:
        def forecast(self,scene,edits):return cached[scene.parent_id]
    record['physical']=measure(Cached(),data_folder,'validation')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/contact-learning'));p.add_argument('--data',type=Path,default=Path('data/architecture/validation'));p.add_argument('--cells',nargs='+',default=['P0','P1','P2','P3','P4']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);a=p.parse_args()
    for seed in a.seeds:
        for cell in a.cells:
            cp=a.root/cell/f'seed-{seed}'/'model.pt';out=cp.parent/'evaluation.json'
            if out.exists():
                if json.loads(out.read_text())['checkpoint_sha256']!=digest(cp):raise ValueError('stale evaluation')
                continue
            result=evaluate(cp,a.data);out.write_text(json.dumps(result,indent=2));print(cell,seed,result['metrics']['release']['all_branches']['raw_contact'],flush=True)
