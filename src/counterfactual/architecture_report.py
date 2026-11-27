"""Static factorial evidence, regenerated only from saved numerical records."""
import argparse
import json
from pathlib import Path
import numpy as np
from .figures import style,save,scene_axes,BLUE,ORANGE,GREEN,RUST,GREY
import matplotlib.pyplot as plt
from .evaluation import interval

COLORS=dict(A=BLUE,B=ORANGE,C=GREEN,D=RUST,analytic=GREY)
LABELS=['A\ndirect','B\ndifference','C\ndirect + JE','D\ndifference + JE','Analytic']


def plot_context(root):
    context_path=root/'context-diagnostics.json'
    if context_path.exists():
        examples=json.loads(context_path.read_text())['rows'];fig,axes=plt.subplots(1,3,figsize=(13.4,3.5),layout='constrained')
        for entry,color in zip(examples,[BLUE,ORANGE]):
            axes[0].plot(entry['context_times_s'],entry['context_difference'],color=color,label=entry['label'])
        axes[0].set(xlabel='Time after release (s)',ylabel='Interaction-context difference (internal units)',title='(a) Anticipation can change immediately');axes[0].legend(fontsize=10)
        for ax,entry,title in zip(axes[1:],examples,['(b) Physical influence waits','(c) An irrelevant edit stays local']):
            ax.plot(entry['times_s'],entry['actual_effect_radii'],color=GREY,label='Simulator',lw=2)
            ax.plot(entry['times_s'],entry['predicted_effect_radii'],color=GREEN,ls='--',label='D prediction')
            if entry['contact_time_s'] is not None:
                ax.axvline(entry['contact_time_s'],ls=':',color=RUST,label='First bumper contact')
            ax.set(xlabel='Time after release (s)',ylabel='Position effect (ball radii)',title=title);ax.legend(fontsize=10)
        for ax in axes:ax.grid(True)
        save(fig,root,'architecture-context')


def plot_phase_audit(root):
    path=root/'phase-audit.json'
    if not path.exists():return
    record=json.loads(path.read_text());fig,axes=plt.subplots(1,2,figsize=(10.6,3.6),layout='constrained')
    for i,v in enumerate('ABCD'):
        rows=[record['models'][f'{v}/seed-{s}'] for s in range(3)]
        axes[0].scatter(i+np.linspace(-.1,.1,3),[r['contact']['f1'] for r in rows],color=COLORS[v],s=30)
    axes[0].scatter(4,record['geometric_admission']['f1'],color=GREY,marker='s',s=30)
    axes[0].set(xticks=range(5),xticklabels=['A','B','C','D','Geometry'],ylim=(-.04,1.04),ylabel='Contact interval F1',title='(a) Exact states do not repair the readout')
    edges=np.array(record['histogram_edges'])
    for key,color,ls,label in [('contact',GREEN,'-','Actual contact'),('no_contact',GREY,'--','No contact')]:
        counts=np.array([r['scores'][key]['histogram'] for r in record['models'].values()]).sum(0)
        axes[1].stairs(counts/counts.sum(),edges,color=color,ls=ls,label=label,lw=2)
    axes[1].axvline(.5,color=RUST,ls=':',label='Readout threshold')
    axes[1].set(xlabel='Contact score on exact states',ylabel='Fraction within each actual class',title='(b) Low scores suppress event reports');axes[1].legend(fontsize=10)
    for ax in axes:ax.grid(True)
    save(fig,root,'architecture-contact-audit')


def build(root):
    style();dec=json.loads((root/'decisions.json').read_text());cal=json.loads((root/'calibration.json').read_text())
    training={str(p.parent.relative_to(root)):json.loads(p.read_text()) for p in sorted(root.glob('*/seed-*/training.json'))}
    physical={name:json.loads((root/'physical'/f"{name.replace('/','-')}.json").read_text()) for name in dec['models']}
    contrasts={};rows={mode:{} for mode in ['fixed','calibrated']}
    for mode in rows:
        for variant in 'ABCD':rows[mode][variant]=[dict(r,seed=seed) for seed in range(3) for r in dec['models'][f'{variant}/seed-{seed}'][mode]['rows']]
        rows[mode]['analytic']=dec['models']['analytic'][mode]['rows']
        contrasts[mode]={}
        for after,before in [('B','A'),('D','C'),('C','A'),('D','B')]:
            left={(r['parent_id'],r['goal_index'],r['seed']):r for r in rows[mode][before]}
            delta=[dict(parent_id=r['parent_id'],difference=int(r['success'])-int(left[(r['parent_id'],r['goal_index'],r['seed'])]['success'])) for r in rows[mode][after] if r['solvable']]
            contrasts[mode][f'{after}-{before}']=interval(delta,lambda r:r['difference'],lambda r:1)
    summaries={}
    for variant in [*'ABCD','analytic']:
        names=[f'{variant}/seed-{s}' for s in range(3)] if variant!='analytic' else ['analytic']
        summaries[variant]=dict(
            seeds=[dict(name=name,threshold_m=cal['selected_thresholds_m'][name],
              fixed=dec['models'][name]['fixed']['summary'],calibrated=dec['models'][name]['calibrated']['summary'],
              test_position_radii=float(np.mean([r['position_mean_radii'] for r in physical[name]['test']['parents']])),
              test_difference_radii=float(np.mean([r['difference_radii'] for r in physical[name]['test']['parents']])),
              no_effect=physical[name]['test']['no_effect'],pre_influence=physical[name]['test']['pre_influence'],
              no_effect_max_radii=max([x for p in physical[name]['test']['parents'] for x in p['no_effect']],default=0),
              pre_influence_max_radii=max([x for p in physical[name]['test']['parents'] for x in p['pre']],default=0),
              contact_f1={k:v['f1'] for k,v in physical[name]['test']['contact'].items()},
              cold_forecast_ms=float(np.median([r['cold_forecast_s'] for r in dec['latencies'][name]]))*1000,
              parameters=training[name]['parameters'] if name in training else 0,
              inactive_auxiliary_parameters=training[name]['inactive_auxiliary_parameters'] if name in training else 0,
              training_seconds=training[name]['seconds'] if name in training else 0,
              selected_embedding_diagnostics=min(training[name]['history'],key=lambda h:h['validation_vector_rmse_m'])['embedding'] if name in training else None,
              validation_selected_update=min(training[name]['history'],key=lambda h:h['validation_vector_rmse_m'])['update'] if name in training else None)
              for name in names])
    summary=dict(protocol=json.loads((root/'protocol.json').read_text()),variants=summaries,paired_success_contrasts=contrasts,
                 uncertainty='2000 paired parent bootstrap resamples; all goals/branches and three fixed training seeds stay together per parent. Intervals are unadjusted for the four declared contrasts and do not estimate population training-seed variance or calibration-fit uncertainty.',
                 promotion='No automatic deployment. Compare analytic priors and report limits before attributing gains to learning.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13.4,3.7),layout='constrained')
    for i,v in enumerate(summaries):
        rr=summaries[v]['seeds'];x=i+np.linspace(-.10,.10,len(rr)) if len(rr)>1 else [i]
        axes[0].scatter(x,[r['fixed']['success_solvable']['estimate'] for r in rr],color=COLORS[v],marker='s' if v=='analytic' else 'o',s=30)
        axes[1].scatter(x,[r['test_difference_radii'] for r in rr],color=COLORS[v],marker='s' if v=='analytic' else 'o',s=30)
        axes[2].scatter(x,[r['contact_f1']['ramp'] for r in rr],color=COLORS[v],marker='s' if v=='analytic' else 'o',s=30)
    for ax,title,label in zip(axes,['(a) Verified repairs','(b) Intervention-effect error','(c) Ramp event readout'],['Success / grid-solvable tasks','Paired difference error (ball radii)','Contact interval F1']):
        ax.set(xticks=range(5),xticklabels=LABELS,title=title,ylabel=label);ax.tick_params(axis='x',labelsize=10);ax.grid(True)
    axes[0].set_ylim(0,1.03);axes[2].set_ylim(-.04,1.04)
    axes[2].text(.04,.55,'A–D: learned phase ≥ 0.5\nAnalytic: geometric admission',transform=axes[2].transAxes,fontsize=10,color=GREY)
    save(fig,root,'architecture-comparison')
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.6),layout='constrained')
    for i,(name,result) in enumerate(contrasts['fixed'].items()):
        mean=result['estimate']*100;lo,hi=np.array(result['ci95'])*100
        axes[0].errorbar(i,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',color=[ORANGE,RUST,GREEN,RUST][i],capsize=3)
    axes[0].axhline(0,color=GREY,lw=1,ls=':');axes[0].set(xticks=range(4),xticklabels=['B − A','D − C','C − A','D − B'],ylabel='Verified success change (percentage points)',title='(a) Separate the two hypotheses')
    for v,s in summaries.items():
        rr=s['seeds'];axes[1].scatter([r['cold_forecast_ms'] for r in rr],[r['test_difference_radii'] for r in rr],color=COLORS[v],marker='s' if v=='analytic' else 'o',label=v)
    axes[1].set(xlabel='Cold 257-candidate forecast (ms)',ylabel='Paired difference error (ball radii)',title='(b) Accuracy and runtime');axes[1].legend(ncol=3)
    for ax in axes:ax.grid(True)
    save(fig,root,'architecture-effects')
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.7),layout='constrained')
    for v in summaries:
        points=[]
        for mode,marker in [('fixed','o'),('calibrated','^')]:
            r=rows[mode][v];coverage=interval(r,lambda t:t['selected'],lambda t:1);risk=interval(r,lambda t:t['selected'] and not t['success'],lambda t:t['selected'])
            if risk['estimate'] is None:continue
            x,y=coverage['estimate'],risk['estimate'];lo,hi=risk['ci95'];points.append((x,y))
            axes[0].errorbar(x,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt=marker,color=COLORS[v],capsize=2,label=v if mode=='fixed' else None)
        if len(points)==2:axes[0].plot(*zip(*points),color=COLORS[v],lw=1,alpha=.6)
    axes[0].axhline(.1,color=GREY,ls=':',lw=1);axes[0].set(xlabel='Proposal coverage',ylabel='Failed / proposed',title='(a) Fixed ○ and calibrated △ margins');axes[0].legend(ncol=3)
    for i,v in enumerate(summaries):
        rr=summaries[v]['seeds'];success=np.mean([r['calibrated']['successes'] for r in rr]);failure=np.mean([r['calibrated']['failures'] for r in rr])
        axes[1].bar(i,success,color=GREEN,width=.58,label='Verified successes' if i==0 else None);axes[1].bar(i,failure,bottom=success,color=RUST,width=.58,label='Failed proposals' if i==0 else None)
    axes[1].set(xticks=range(5),xticklabels=LABELS,ylabel='Mean tasks per seed',title='(b) Count failures alongside successes',ylim=(0,220));axes[1].tick_params(axis='x',labelsize=10);axes[1].legend(loc='upper center',ncol=2,fontsize=10)
    for ax in axes:ax.grid(True,axis='y');ax.set_axisbelow(True)
    save(fig,root,'architecture-calibration')
    fig,axes=plt.subplots(1,3,figsize=(13.4,3.5),layout='constrained')
    for v in 'ABCD':
        histories=[training[f'{v}/seed-{s}']['history'] for s in range(3)];x=[h['update'] for h in histories[0]];values=np.array([[h['validation_vector_rmse_m'] for h in hist] for hist in histories])
        axes[0].plot(x,values.mean(0),'-o',color=COLORS[v],label=v);axes[0].fill_between(x,values.min(0),values.max(0),color=COLORS[v],alpha=.09)
        if v in 'CD':
            for ax,key in zip(axes[1:],['mean_std','effective_rank']):
                z=np.array([[h['embedding'][key] for h in hist] for hist in histories]);ax.plot(x,z.mean(0),'-o',color=COLORS[v],label=v);ax.fill_between(x,z.min(0),z.max(0),color=COLORS[v],alpha=.12)
    for ax,title,label in zip(axes,['(a) Physical validation','(b) Embedding spread','(c) Embedding effective rank'],['Vector-position RMSE (m)','Mean coordinate standard deviation','Effective rank / 16 dimensions']):
        ax.set(xlabel='Optimizer update',ylabel=label,title=title);ax.grid(True);ax.legend()
    axes[0].set_yscale('log');axes[2].set_ylim(0,16.5);axes[2].axhline(16,color=GREY,ls=':',lw=1)
    save(fig,root,'architecture-learning')
    plot_context(root)
    plot_phase_audit(root)
    gallery=json.loads((root/'gallery.json').read_text())
    for category in ['success','failure']:
        if category not in gallery:continue
        g=gallery[category];fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained');scene_axes(axes[0],g)
        for key,color,ls,label in [('original',GREY,'-','Original'),('verified',GREEN,'-','Verified edit'),('predicted',BLUE,'--','A prediction')]:
            p=np.array(g[key]);axes[0].plot(p[:,0],p[:,1],color=color,ls=ls,label=label)
        axes[0].set_title(f'(a) First A seed-0 {category}');axes[0].legend(fontsize=10)
        grid=np.array(g['grid'])/g['scene']['ball']['radius'];axes[1].plot(grid,g['actual_margins'],color='#303b40',label='Simulator',lw=2)
        for name,margins in g['predicted_margins'].items():
            v=name.split('/')[0];axes[1].plot(grid,margins,color=COLORS[v],ls='--',label=v,lw=1.3)
        axes[1].axhline(0,color=GREY,lw=1);axes[1].axhline(.03,color=GREY,ls=':',lw=1);axes[1].axvline(g['edit_m']/g['scene']['ball']['radius'],color=BLUE,ls=':',lw=1)
        axes[1].set(xlabel='Displacement / ball radius',ylabel='Goal margin (m)',title='(b) Full edit landscape');axes[1].grid(True);axes[1].legend(ncol=3,fontsize=9)
        chosen=int(np.abs(np.array(g['grid'])-g['edit_m']).argmin());near=slice(max(0,chosen-3),min(len(grid),chosen+4))
        for values,color,ls,label in [(g['actual_margins'],GREEN,'-','Simulator'),(g['predicted_margins']['A/seed-0'],BLUE,'--','A prediction')]:
            axes[2].plot(grid[near],np.array(values)[near],ls=ls,marker='o',ms=3,color=color,label=label)
            axes[2].scatter(grid[chosen],values[chosen],color=color,s=55,zorder=4)
        axes[2].axhline(0,color=GREY,lw=1);axes[2].axhline(.03,color=GREY,ls=':',lw=1);axes[2].axvline(grid[chosen],color=GREY,ls=':',lw=1)
        axes[2].set(xlabel='Displacement / ball radius',ylabel='Goal margin (m)',title='(c) The chosen boundary');axes[2].grid(True);axes[2].legend(fontsize=10)
        save(fig,root,f'architecture-{category}')
    if 'contact_transition' in gallery:
        g=gallery['contact_transition'];fig,axes=plt.subplots(1,2,figsize=(10.4,3.5),layout='constrained');grid=np.array(g['grid'])/g['scene']['ball']['radius'];valid=np.array(g['valid'])
        axes[0].step(grid,np.where(valid,np.array(g['contact'],int),np.nan),where='mid',color=GREY);axes[0].set(xlabel='Bumper displacement / ball radius',ylabel='Bumper contact occurred',yticks=[0,1],title='(a) An edit can create an encounter')
        axes[1].plot(grid,g['final_dx'],color='#303b40',label='Simulator')
        for name,dx in g['predicted_final_dx'].items():
            v=name.split('/')[0];axes[1].plot(grid[valid],dx,color=COLORS[v],ls='--',label=v)
        axes[1].set(xlabel='Bumper displacement / ball radius',ylabel='Final horizontal effect (m)',title='(b) Small edit need not mean small effect');axes[1].legend(ncol=3,fontsize=9)
        for ax in axes:ax.grid(True)
        save(fig,root,'architecture-contact-boundary')
    print(json.dumps(dict(contrasts=contrasts,variants={k:[dict(fixed_success=r['fixed']['successes'],calibrated_success=r['calibrated']['successes'],calibrated_failures=r['calibrated']['failures']) for r in v['seeds']] for k,v in summaries.items()}),indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/architecture'));p.add_argument('--context-only',action='store_true');a=p.parse_args();style();plot_context(a.root) if a.context_only else build(a.root)
