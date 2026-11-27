"""Summarize paired local predictions and draw the measured contact witness."""
import argparse
import json
from pathlib import Path

import numpy as np
from .architecture_data import digest
from .competition_scenes import DATA, PROTOCOL, REPORT
from .evaluation import interval
from .figures import BLUE, GREEN, ORANGE, RUST, GREY, save, style
from .geometry import segment_distance
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def read(path):
    return json.loads(path.read_text())


def rank_credit(winner, loser):
    return .5 if abs(winner-loser)<1e-6 else float(winner>loser)


def summarize(data=DATA, output=REPORT):
    config=read(PROTOCOL)
    splits={name:read(data/name/'manifest.json') for name in config['splits']}
    compact={}
    parent_losses={}
    for cell in ['E1','E2']:
        compact[cell]=[]
        for seed in config['comparison']['seeds']:
            run=read(output/cell/f'seed-{seed}'/'evaluation.json')
            rows=run['parents']
            legal=[r for r in rows if r['legal_edit'] is not None]
            if run['binding']['manifest_sha256']['validation'] != digest(data/'validation/manifest.json'):
                raise ValueError('Validation manifest mismatch')
            for r in rows:
                parent_losses.setdefault(r['parent_id'],{})[(cell,seed)]=r['bce']
            compact[cell].append(dict(seed=seed,bce=float(np.mean([r['bce'] for r in rows])),
                winner_accuracy=float(np.mean([rank_credit(r['winner_probability'],r['loser_probability']) for r in rows])),
                intact_mean=float(np.mean([r['loser_probability'] for r in rows])),
                isolated_mean=float(np.mean([r['recovered_probability'] for r in rows])),
                legal_parents=len(legal),
                legal_threshold_flips=sum(r['loser_probability']<.5<=r['legal_edit']['ramp_probability'] for r in legal),
                legal_scores=[dict(parent_id=r['parent_id'],intact=r['loser_probability'],
                                   edited=r['legal_edit']['ramp_probability']) for r in legal],
                legal_gain=float(np.mean([r['legal_edit']['ramp_probability']-r['loser_probability'] for r in legal])) if legal else None,
                legal_rank=float(np.mean([rank_credit(r['legal_edit']['ramp_probability'],r['loser_probability']) for r in legal])) if legal else None,
                ordinary=run['ordinary']))
    differences=[dict(parent_id=k,delta=np.mean([v['E2',s]-v['E1',s] for s in config['comparison']['seeds']]))
                 for k,v in parent_losses.items()]
    validation=splits['validation']['records']
    controls={}
    for name,key in [('nearest','gaps_m'),('earliest','times_s')]:
        successes=[]
        for r in validation:
            w=r['audits']['1920']
            a,b=[w[key][w['candidate_ids'].index(surface)] for surface in [w['winner'],w['omitted']]]
            successes.append(rank_credit(-a,-b))
        controls[name]=dict(mean=float(np.mean(successes)),credit=sum(successes),parents=len(successes))
    # First nearest-clearance counterexample in validation order, independent of model outputs.
    example=next(r for r in validation if all(w['nearest']==w['omitted'] for w in r['audits'].values()))
    return dict(kind='competition-scenes-v2',protocol_sha256=digest(PROTOCOL),
                support={name:x['statistics'] for name,x in splits.items()},runs=compact,
                controls=controls,paired_bce=interval(differences,lambda r:r['delta'],lambda r:1),
                example=dict(scene=example['scene'],audits=example['audits']))


def plot_witness(figures,output):
    scene=figures['example']['scene']
    w=figures['example']['audits']['1920']
    fig,axes=plt.subplots(1,3,figsize=(13.4,3.65),layout='constrained')
    ax=axes[0]
    center=scene['bumper']['position']
    ax.add_patch(Circle(center,scene['bumper']['radius'],color=ORANGE,alpha=.42))
    ramp=next(s for s in scene['segments'] if s['id']=='ramp')
    a,b=np.array(ramp['a']),np.array(ramp['b'])
    normal=np.array([a[1]-b[1],b[0]-a[0]])/np.linalg.norm(b-a)
    corners=np.array([a,b,b,a])+np.array([1,1,-1,-1])[:,None]*normal*ramp['radius']
    ax.fill(*corners.T,color=GREY)
    for end in (a,b):
        ax.add_patch(Circle(end,ramp['radius'],color=GREY))
    start=w['interval_start'][:2]
    ax.add_patch(Circle(start,scene['ball']['radius'],fill=False,edgecolor=BLUE,lw=1.5,label='Ball at interval start'))
    ax.plot(start[0],start[1],'o',ms=3,color=BLUE)
    ax.set(xlim=(start[0]-.63,start[0]+.63),ylim=(start[1]-.55,start[1]+.35),
           xlabel='Horizontal position (m)',ylabel='Height (m)',title='(a) Two reachable surfaces',aspect='equal')
    ax.legend(fontsize=9,loc='upper left')
    ax.annotate('',xy=np.array(start)+np.array(w['interval_start'][2:4])*.010,xytext=start,arrowprops=dict(arrowstyle='->',color=BLUE))
    for surface,ax in zip([w['winner'],w['omitted']],axes[1:]):
        for key,label,color,ls in [('observed_xy','Both surfaces',BLUE,'-'),('isolated_xy','Winner removed',RUST,'--')]:
            # Distance to the removed surface itself has no mechanical meaning.
            if surface==w['winner'] and key=='isolated_xy':
                continue
            xy=np.asarray(w[key])
            if surface=='ramp':
                gap=segment_distance(xy,a,b)-ramp['radius']
            else:
                gap=np.linalg.norm(xy-center,axis=1)-scene['bumper']['radius']
            gap=(gap-scene['ball']['radius'])*1000
            ax.plot(np.arange(len(xy))*1000/1920,gap,ls,color=color,label=label)
        time=w['begin_s'] if surface==w['winner'] else w['isolated_begin_s']
        ax.axvline((time-w['tick']/60)*1000,ls=':',color=GREY,lw=1,label='Begin callback')
        ax.axhline(0,color=GREY,lw=.8)
        ax.set(xlabel='Time into interval (ms)',ylabel=f'Ball–{surface} clearance (mm)',
               title=f'(b) {surface.capitalize()} makes contact' if surface==w['winner'] else f'(c) Removing the {w["winner"]} reveals contact')
        ax.grid(True,alpha=.35)
        ax.legend(fontsize=9)
    save(fig,output,'competition-witness')


def plot_evidence(summary,output):
    fig,axes=plt.subplots(1,3,figsize=(13.4,3.65),layout='constrained')
    for name,color,marker in [('E1',BLUE,'o'),('E2',GREEN,'s')]:
        group=summary['runs'][name]
        for seed,r in enumerate(group):
            axes[0].plot(seed,r['bce'],marker,color=color,label=name if seed==0 else None)
            axes[1].plot([0,1],[r['intact_mean'],r['isolated_mean']],ls='-' if name=='E2' else '--',marker=marker,color=color,alpha=.8)
            axes[2].plot(seed,r['winner_accuracy']*100,marker,color=color)
    for index in range(3):
        axes[0].plot([index,index],[summary['runs'][name][index]['bce'] for name in ['E1','E2']],color=GREY,lw=.8,zorder=0)
    axes[0].set(xticks=range(3),xlabel='Training seed',ylabel='Binary cross-entropy (nats)',title='(a) Predicting both candidates')
    axes[0].legend()
    axes[1].set(xticks=[0,1],xticklabels=['Both present','Winner removed'],ylabel='Mean score for preempted surface',ylim=(-.03,1.03),title='(b) Same surface, changed context')
    for control,ls in [('nearest',':'),('earliest','--')]:
        axes[2].axhline(summary['controls'][control]['mean']*100,ls=ls,color=GREY,label=f'{control.capitalize()} surface')
    axes[2].set(xticks=range(3),xlabel='Training seed',ylabel='Correct surface ranked first (%)',title='(c) Geometric controls',ylim=(0,105))
    axes[2].legend(fontsize=9,loc='lower right')
    for ax in axes:ax.grid(True,axis='y',alpha=.35)
    save(fig,output,'competition-readout')


def plot_legal_edits(summary,output):
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout='constrained',sharey=True)
    for cell,ax in zip(['E1','E2'],axes):
        ax.axvspan(0,.5,ymin=.5,ymax=1,color=GREEN,alpha=.09)
        ax.plot([0,1],[0,1],color=GREY,ls=':',lw=1)
        ax.axvline(.5,color=GREY,lw=.8)
        ax.axhline(.5,color=GREY,lw=.8)
        for run,marker,color in zip(summary['runs'][cell],['o','^','s'],[BLUE,ORANGE,GREEN]):
            points=run['legal_scores']
            ax.scatter([p['intact'] for p in points],[p['edited'] for p in points],s=21,
                       marker=marker,color=color,alpha=.75,label=f"Seed {run['seed']}")
        ax.set(xlabel='Ramp begin score before nudge',xlim=(0,1.01),ylim=(0,1.01),
               title='(a) Own surface and history' if cell=='E1' else '(b) With competing-surface times')
        ax.set_aspect('equal')
        ax.grid(True,alpha=.25)
    axes[0].set_ylabel('Ramp begin score after 15 mm nudge')
    axes[0].legend(fontsize=9,loc='upper left')
    axes[1].text(.52,.02,'At 1920 Hz, all 15 pairs\nmiss → contact',va='bottom',fontsize=10)
    save(fig,output,'competition-edits')


def draw(output):
    summary=read(output/'summary.json')
    style()
    plot_witness(summary,output)
    plot_evidence(summary,output)
    plot_legal_edits(summary,output)
    return {'figures':['competition-witness','competition-readout','competition-edits']}


def draw_continuation(output):
    pairs = read(output / "diagnosis.json")["pairs"]
    style()
    fig, (scores, times) = plt.subplots(1, 2, figsize=(10.8, 5.5), sharey=True, layout="constrained")
    singles = [p for p in pairs if len(p["begins_1920"]) == 1]
    doubles = [p for p in pairs if len(p["begins_1920"]) > 1]
    rows, labels = [], []
    y = 0
    for group, name in [(singles, "Ramp alone"), (doubles, "Two surfaces")]:
        scores.text(.005, y, f"{name}  ({len(group)} parents)", fontsize=11)
        for p in sorted(group, key=lambda p: p["parent_id"]):
            y += 1
            rows.append(y)
            labels.append(p["parent_id"].rsplit("-", 1)[-1])
            s = p["scores"]["0"]
            scores.plot([s["original"], s["edited"]], [y, y], color=BLUE, lw=1.2)
            scores.plot(s["original"], y, "o", mfc="white", mec=BLUE, ms=5, zorder=3)
            scores.plot(s["edited"], y, "o", color=BLUE, ms=4, zorder=4)
            arrivals = [next(g["ms"] for g in p[k] if "ramp" in g["surfaces"])
                        for k in ("begins_1920", "begins_3840_two_intervals")]
            times.plot(arrivals, [y, y], color=GREY, lw=1.1)
            times.plot(arrivals[0], y, "s", mfc="white", mec=GREY, ms=6)
            times.plot(arrivals[1], y, "o", color=GREEN, ms=3.9)
        y += 1.6
    scores.plot([], [], "o", mfc="white", mec=BLUE, ms=5, label="Original release")
    scores.plot([], [], "o", color=BLUE, ms=4, label="Edited release")
    times.plot([], [], "s", mfc="white", mec=GREY, ms=6, label="1920 Hz")
    times.plot([], [], "o", color=GREEN, ms=4, label="3840 Hz")
    scores.axvline(.5, color=GREY, ls=":", lw=1)
    scores.text(.52, 0, "Cutoff 0.5", color=GREY, fontsize=10)
    frame_ms = 1000 / 60
    times.axvspan(frame_ms, 24, color=GREY, alpha=.065, lw=0)
    times.axvline(frame_ms, color=GREY, ls=":", lw=1)
    times.text(frame_ms + .4, 0, "Next interval", color=GREY, fontsize=10)
    scores.set(xlim=(-.025, 1.025), xlabel="Predicted ramp begin score (E2, seed 0)",
               yticks=rows, yticklabels=labels, ylim=(rows[-1] + .8, -1), ylabel="Parent scene")
    times.set(xlim=(0, 24), xticks=range(0, 25, 4), xlabel="Edited ramp begin after interval start (ms)")
    for ax, title in [(scores, "(a) Does the nudge cross the cutoff?"),
                      (times, "(b) When does the ramp contact begin?")]:
        ax.set_title(title, loc="left", pad=28)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, fontsize=10,
                  handlelength=1, borderaxespad=0)
        ax.grid(axis="x", alpha=.32)
        ax.tick_params(axis="y", length=0, labelsize=10)
    save(fig, output, "ramp-continuation")
    return {"figures": ["ramp-continuation"]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPORT)
    args=parser.parse_args()
    result=summarize(output=args.output)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    (args.output/'report-kind.json').write_text(json.dumps({'kind':result['kind']}))
    print(draw(args.output))
