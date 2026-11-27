"""Rebuild the controlled experiment summary and static figures from records."""
import json
from pathlib import Path
import numpy as np
from .figures import style,save,BLUE,GREEN,ORANGE,RUST,GREY
import matplotlib.pyplot as plt


def report(root=Path('artifacts/mechanics')):
    style();rows=[];dec=json.loads((root/'decisions.json').read_text())
    for cp in sorted(root.glob('*/seed-*/training.json')):
        train=json.loads(cp.read_text());diag=json.loads((cp.parent/'diagnostics.json').read_text())
        v=diag['validation'];s=dec['models'][str(cp.parent.relative_to(root))]['summary']
        rows.append(dict(cell=train['cell'],seed=train['seed'],parameters=train['parameters'],processed_intervals=train['processed_training_intervals'],train_seconds=train['seconds'],validation_rmse_m=min(h['validation_vector_rmse_m'] for h in train['history']),pre=v['pre_influence'],no_effect=v['no_effect'],ramp_f1=v['contact']['ramp']['f1'],difference_radii=float(np.mean([p['difference_radii'] for p in v['parents']])),zero_difference_radii=float(np.mean([p['zero_effect_predictor_error_radii'] for p in v['parents']])),ramp_error_radii=v['families']['ramp']['position_mean_radii'],ramp_tail_radii=v['families']['ramp']['position_p95_radii'],decisions=s))
    cells=['soft-window','soft-release','swept-window','swept-release','swept-local-release']
    summaries={c:{key:[r[key] for r in rows if r['cell']==c] for key in ['validation_rmse_m','parameters','processed_intervals','train_seconds','ramp_f1','difference_radii']} for c in cells}
    selected=min(cells,key=lambda c:np.mean(summaries[c]['validation_rmse_m']))
    ref=[json.loads((root/'references'/f'{n}.json').read_text())['validation'] for n in ['E0','A']]
    gates={}
    for c in cells:
        local=[r for r in rows if r['cell']==c]
        gates[c]=dict(locality=all(r[k][metric]<min(v['pre_influence' if k=='pre' else 'no_effect'][metric] for v in ref) for r in local for k in ['pre','no_effect'] for metric in ['mean','p95']),sensitivity=all(r['difference_radii']<r['zero_difference_radii'] for r in local),ramp=all(r['ramp_f1']>=ref[0]['contact']['ramp']['f1'] for r in local))
    summary=dict(rows=rows,cells=summaries,validation_selected_cell=selected,representative_seed=0,gates=gates,scope='development cohort; three training seeds, not independent final tests; revised mechanics are a bundle of physical priors')
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    labels=['Soft\nwindow','Soft\nrelease','Swept\nwindow','Swept\nrelease','Swept local\nrelease'];colors=[GREY,ORANGE,BLUE,GREEN,RUST]
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.7),layout='constrained')
    for i,c in enumerate(cells):
        rr=[r for r in rows if r['cell']==c];x=i+np.linspace(-.12,.12,3)
        axes[0].scatter(x,[r['decisions']['successes']/28 for r in rr],color=colors[i],s=25)
        axes[1].scatter(x,[r['no_effect']['mean'] for r in rr],color=colors[i],s=25)
        axes[2].scatter(x,[r['ramp_f1'] for r in rr],color=colors[i],s=25)
    titles=['(a) Verified repairs','(b) Spurious edit effects shrink','(c) Ramp timing recovers']
    ylabs=['Success / 28 grid-solvable tasks','Mean no-effect error (ball radii)','Ramp interval F1']
    for ax,title,ylabel in zip(axes,titles,ylabs):
        ax.set(xticks=range(5),xticklabels=labels,title=title,ylabel=ylabel);ax.grid(True);ax.tick_params(axis='x',labelsize=10)
    axes[0].set_ylim(0,1);axes[2].set_ylim(0,1.05)
    save(fig,root,'mechanics-comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for i,c in enumerate(cells):
        rr=[r for r in rows if r['cell']==c]
        axes[0].scatter([r['decisions']['proposals'] for r in rr],[r['decisions']['failed_proposals'] for r in rr],s=32,color=colors[i],label=c)
        axes[1].scatter([r['processed_intervals'] for r in rr],[r['validation_rmse_m'] for r in rr],s=32,color=colors[i],label=c)
    axes[0].set(xlabel='Proposals / 68 failing tasks',ylabel='Failed proposals',title='(a) Count failures alongside coverage')
    axes[1].set(xlabel='Processed training intervals',ylabel='Validation vector RMSE (m)',title='(b) Longer history has a compute cost',yscale='log')
    axes[1].set_xticks([51200,384000],['51,200','384,000'])
    axes[0].legend(fontsize=9)
    for ax in axes:ax.grid(True)
    save(fig,root,'mechanics-budgets')
    calibration=root/'calibration/calibration.json'
    if calibration.exists():
        cal=json.loads(calibration.read_text());fit=cal['fit'];threshold=[float(k) for k in fit];fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
        axes[0].plot([fit[str(t)]['coverage'] for t in threshold],[fit[str(t)]['false_success'] for t in threshold],'o-',color=BLUE,label='Calibration fit')
        audit=cal['audit'];lo,hi=audit['false_success_ci']['ci95'];v=audit['false_success']
        axes[0].errorbar(audit['coverage'],v,yerr=[[v-lo],[hi-v]],fmt='s',color=GREEN,capsize=3,label='Independent audit · 95% CI');axes[0].axhline(.1,ls=':',color=GREY)
        axes[0].set(xlabel='Selection coverage',ylabel='Failed / proposed',title='(a) Calibrate after candidate selection');axes[0].legend()
        for key,color in [('successes',GREEN),('failures',RUST),('proposals',BLUE)]:axes[1].plot(threshold,[fit[str(t)][key] for t in threshold],'o-',color=color,label=key.capitalize())
        axes[1].set(xlabel='Required predicted margin (m)',ylabel='Task count',title='(b) The cost of a stricter margin');axes[1].legend()
        for ax in axes:ax.grid(True)
        save(fig,root,'selection-calibration')
    print(json.dumps(dict(selected=selected,gates=gates),indent=2))

if __name__=='__main__':report()
