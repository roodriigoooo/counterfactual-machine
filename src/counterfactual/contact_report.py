"""A-only diagnosis, causal episode context, integration and tolerance evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from .figures import style,save,scene_axes,BLUE,ORANGE,GREEN,RUST,GREY
import matplotlib.pyplot as plt
from .evaluation import interval


def read(path):return json.loads(Path(path).read_text())

def summarize_rows(rows):
    proposals=[r for r in rows if r['selected']];repairs=[r for r in rows if not r['initial_success']]
    return dict(tasks=len(rows),parents=len({r['parent_id'] for r in rows}),proposals=len(proposals),center_successes=sum(r['center_success'] for r in rows),perturbed_successes=sum(r['perturbed_success'] for r in rows),
        coverage=interval(rows,lambda r:r['selected'],lambda r:1),perturbed_success=interval(rows,lambda r:r['perturbed_success'],lambda r:1),
        conditional_failure=interval(rows,lambda r:r['selected'] and not r['perturbed_success'],lambda r:r['selected']),
        mean_edit_mm=float(np.mean([abs(r['edit_m'])*1000 for r in proposals])) if proposals else None,
        mean_sampled_success=float(np.mean([r['perturbation_success_fraction'] for r in proposals])) if proposals else None,
        worst_margin_quantiles_m=np.quantile([r['worst_margin_m'] for r in proposals if r['worst_margin_m'] is not None],[0,.05,.5]).tolist() if any(r['worst_margin_m'] is not None for r in proposals) else None,
        boundary_error_mm={k:float(v) for k,v in zip(['median','p95'],np.quantile([r['boundary_error_mm'] for r in proposals if r.get('boundary_error_mm') is not None],[.5,.95]))} if any(r.get('boundary_error_mm') is not None for r in proposals) else None)


def training_figures(output):
    cells=['P0','P1','P2','P3','P4'];colors=[GREY,BLUE,ORANGE,GREEN,RUST];records={}
    for cell in cells:
        records[cell]=[dict(training=read(f'artifacts/contact-learning/{cell}/seed-{seed}/training.json'),evaluation=read(f'artifacts/contact-learning/{cell}/seed-{seed}/evaluation.json')) for seed in range(3)]
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.6),layout='constrained')
    for i,(cell,color) in enumerate(zip(cells,colors)):
        rr=records[cell];x=i+np.linspace(-.1,.1,3)
        axes[0].scatter(x,[r['evaluation']['metrics']['release']['all_branches']['raw_contact']['f1'] for r in rr],color=color,s=28)
        axes[1].scatter(x,[r['training']['history'][-1]['validation_vector_rmse_m'] for r in rr],color=color,s=28)
        axes[2].scatter(x,[r['training']['history'][0]['gradient_norms']['phase']/r['training']['history'][0]['gradient_norms']['physical'] for r in rr],color=color,s=28)
    for ax,title,label in zip(axes,['(a) Learning to report contact','(b) Physical accuracy must survive','(c) The objective reaches the encoder'],['Raw contact F1, full release','Vector-position RMSE (m)','Initial phase / physical gradient norm']):
        ax.set(xticks=range(5),xticklabels=['P0\noriginal','P1\nlonger','P2\nrescaled','P3\nexact states','P4\nbalanced'],title=title,ylabel=label);ax.tick_params(axis='x',labelsize=10);ax.grid(True)
    axes[0].set_ylim(-.03,1.03);axes[2].set_yscale('log');save(fig,output,'contact-training')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),layout='constrained')
    for i,(cell,color) in enumerate(zip(cells,colors)):
        rr=records[cell];x=i+np.linspace(-.1,.1,3)
        axes[0].scatter(x,[r['evaluation']['metrics']['release']['all_branches']['raw_contact']['precision'] or 0 for r in rr],color=color,s=28)
        differences=[r['evaluation']['metrics']['release']['all_branches']['masked_contact']['f1']-r['evaluation']['metrics']['release']['all_branches']['always_positive_masked']['f1'] for r in rr]
        axes[1].scatter(x,np.array(differences)*100,color=color,s=28)
    for ax,title,label in zip(axes,['(a) Recall alone is insufficient','(b) Does the head add to the mask?'],['Raw contact precision (0 = no reports)','Masked F1 − always-positive masked (pp)']):
        ax.set(xticks=range(5),xticklabels=cells,title=title,ylabel=label);ax.grid(True)
    axes[1].axhline(0,color=GREY,ls=':',lw=1);save(fig,output,'contact-mask-control')
    return records


def episode_figures(output):
    records={cell:[{str(update):read(f'artifacts/episodes/{cell}/seed-{seed}/update-{update}.json') for update in [200,300]} for seed in range(3)] for cell in ['E0','E1','E2']}
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.6),layout='constrained');modes=['exact_oracle_history','exact_self_history','release']
    for cell,color in zip(records,[BLUE,ORANGE,GREEN]):
        for ax,event in zip(axes,['begin','separate','active_end']):
            values=np.array([[r['300']['metrics'][m]['labels' if event=='active_end' else 'events'][event]['f1'] for m in modes] for r in records[cell]])
            ax.plot(range(3),values.mean(0),'-o',color=color,label=cell);ax.fill_between(range(3),values.min(0),values.max(0),color=color,alpha=.12)
    for ax,title in zip(axes,['(a) Actual begin events','(b) Actual separation events','(c) Contact at interval end']):
        ax.set(xticks=range(3),xticklabels=['Exact state\ntrue history','Exact state\nown history','Free release\nown history'],title=title,ylabel='Event F1' if ax!=axes[2] else 'Active-end interval F1',ylim=(0,1));ax.tick_params(axis='x',labelsize=10);ax.grid(True);ax.legend()
    save(fig,output,'contact-episodes')
    return records


def integrator_figures(output):
    records={str(n):read(f'artifacts/integrator-study/substeps-{n}.json') for n in [1,4,16]};fig,axes=plt.subplots(1,3,figsize=(13.5,3.6),layout='constrained')
    summaries={}
    for i,(key,record) in enumerate(records.items()):
        rows=[r for r in record['rows'] if r['kind']=='suite'];count={k:sum(v[k] for r in rows for v in r['events'].values()) for k in ['matched','invented','missed']}
        summaries[key]=dict(**count,position_mean_radii=float(np.mean([r['position_mean_radii'] for r in rows])),median_seconds=float(np.median([r['seconds'] for r in rows])),multiple_onset_intervals=sum(r['multiple_onset_intervals'] for r in rows))
        axes[0].bar(i-.16,count['matched'],width=.3,color=GREEN,label='Matched' if i==0 else None);axes[0].bar(i+.16,count['missed'],width=.3,color=RUST,label='Missed' if i==0 else None)
        axes[1].bar(i,count['invented'],width=.55,color=ORANGE)
        axes[2].plot(summaries[key]['median_seconds']*1000,summaries[key]['position_mean_radii'],'o',color=[BLUE,GREEN,RUST][i],label=f'{key} subinterval' + ('' if key=='1' else 's'))
    axes[0].set(xticks=range(3),xticklabels=['1','4','16'],xlabel='Subintervals per observation',ylabel='Fixture onsets',title='(a) Resolve more than one encounter');axes[0].legend()
    axes[1].set(xticks=range(3),xticklabels=['1','4','16'],xlabel='Subintervals per observation',ylabel='Unmatched inferred onsets',title='(b) Resolution can expose chatter')
    axes[2].set(xlabel='Median fixture forecast (ms)',ylabel='Mean path error (ball radii)',title='(c) Path error hides the event problem');axes[2].legend(fontsize=10)
    for ax in axes:ax.grid(True,axis='y');ax.set_axisbelow(True)
    save(fig,output,'contact-integration');return summaries


def tolerance_figures(output):
    root=Path('artifacts/tolerance-study');config=read(root/'protocol.json');files=sorted((root/'evaluation-parents').glob('*.json'))
    if len(files)!=config['parent_pools']['evaluation'][1]:raise ValueError('final evaluation incomplete')
    parents=[read(p) for p in files];rows=[r for p in parents for r in p['rows']];summaries={};contrasts={}
    for model in ['analytic','learned-0','learned-1','learned-2']:
        summaries[model]={}
        for mode in ['fixed','calibrated']:
            summaries[model][mode]={}
            for selector in ['nominal','tolerant']:
                group=[r for r in rows if r['model']==model and r['mode']==mode and r['selector']==selector]
                summaries[model][mode][selector]=dict(all=summarize_rows(group),repairs=summarize_rows([r for r in group if not r['initial_success']]))
    for group in ['analytic','learned']:
        contrasts[group]={}
        for mode in ['fixed','calibrated']:
            subset=[r for r in rows if r['mode']==mode and not r['initial_success'] and (r['model']=='analytic' if group=='analytic' else r['model'].startswith('learned'))]
            nominal={(r['parent_id'],r['goal_index'],r['model']):r for r in subset if r['selector']=='nominal'}
            delta=[dict(parent_id=r['parent_id'],delta=int(r['perturbed_success'])-int(nominal[(r['parent_id'],r['goal_index'],r['model'])]['perturbed_success'])) for r in subset if r['selector']=='tolerant']
            contrasts[group][mode]=interval(delta,lambda r:r['delta'],lambda r:1)
    paired_edits={}
    for model in summaries:
        nominal={(r['parent_id'],r['goal_index']):r for r in rows if r['model']==model and r['mode']=='fixed' and r['selector']=='nominal' and r['selected'] and not r['initial_success']}
        pairs=[(nominal[(r['parent_id'],r['goal_index'])],r) for r in rows if r['model']==model and r['mode']=='fixed' and r['selector']=='tolerant' and r['selected'] and not r['initial_success'] and (r['parent_id'],r['goal_index']) in nominal]
        paired_edits[model]=dict(tasks=len(pairs),nominal_mm=float(np.mean([abs(a['edit_m'])*1000 for a,b in pairs])),tolerant_mm=float(np.mean([abs(b['edit_m'])*1000 for a,b in pairs])),extra_mm=float(np.mean([(abs(b['edit_m'])-abs(a['edit_m']))*1000 for a,b in pairs])))
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.6),layout='constrained')
    for model,color in zip(summaries,[GREY,BLUE,ORANGE,GREEN]):
        points=[]
        for i,selector in enumerate(['nominal','tolerant']):
            s=summaries[model]['fixed'][selector]['repairs'];points.append((s['coverage']['estimate'],s['conditional_failure']['estimate']))
            axes[0].plot(i,s['perturbed_success']['estimate'],marker='s' if model=='analytic' else 'o',color=color,label=model if i==0 else None)
            axes[2].plot(i,paired_edits[model][selector+'_mm'],marker='s' if model=='analytic' else 'o',color=color)
        axes[2].plot([0,1],[paired_edits[model][s+'_mm'] for s in ['nominal','tolerant']],color=color,lw=1)
        for ax,key in [(axes[0],'perturbed_success')]:
            v=[summaries[model]['fixed'][s]['repairs'][key] for s in ['nominal','tolerant']];ax.plot([0,1],[x['estimate'] if isinstance(x,dict) else x for x in v],color=color,lw=1)
        axes[1].plot(*zip(*points),color=color,lw=1)
        for (x,y),marker in zip(points,['o','^']):axes[1].plot(x,y,marker=marker,color=color)
    axes[0].set(xticks=[0,1],xticklabels=['Nominal','±5 mm tolerance'],ylabel='All five checks pass / failing tasks',title='(a) Reliability after selection');axes[0].legend(fontsize=9)
    axes[1].set(xlabel='Proposal coverage',ylabel='Any failed check / proposed',title='(b) Nominal ○, tolerant △');axes[1].axhline(.1,color=GREY,ls=':',lw=1)
    axes[2].set(xticks=[0,1],xticklabels=['Nominal','±5 mm tolerance'],ylabel='Mean edit on jointly proposed tasks (mm)',title='(c) The price of room to spare')
    for ax in axes:ax.grid(True)
    save(fig,output,'tolerance-selection')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),layout='constrained')
    for i,group in enumerate(['analytic','learned']):
        for j,mode in enumerate(['fixed','calibrated']):
            result=contrasts[group][mode];mean=result['estimate']*100;lo,hi=np.array(result['ci95'])*100
            axes[0].errorbar(i+(j-.5)*.15,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o' if j==0 else '^',color=GREY if i==0 else BLUE,capsize=3,label=mode if i==0 else None)
    axes[0].set(xticks=[0,1],xticklabels=['Analytic','Learned, three seeds'],ylabel='Perturbed success change (pp)',title='(a) Tolerant − nominal, paired parents');axes[0].axhline(0,color=GREY,ls=':',lw=1);axes[0].legend()
    for i,model in enumerate(summaries):
        for j,selector in enumerate(['nominal','tolerant']):
            s=summaries[model]['calibrated'][selector]['repairs'];good=s['perturbed_successes'];bad=s['proposals']-good
            if not s['proposals']:axes[1].text(i+(j-.5)*.32,2,'0',ha='center',color=GREY,fontsize=10)
            axes[1].bar(i+(j-.5)*.32,good,width=.3,color=GREEN,alpha=1 if j else .45,label='All checks pass' if i==0 and j else None)
            axes[1].bar(i+(j-.5)*.32,bad,bottom=good,width=.3,color=RUST,alpha=1 if j else .45,label='A check fails' if i==0 and j else None)
    axes[1].set(xticks=range(4),xticklabels=['Analytic','Seed 0','Seed 1','Seed 2'],ylabel='Calibrated proposals on failing tasks',title='(b) Nominal left, tolerant right');axes[1].legend(fontsize=10)
    for ax in axes:ax.grid(True,axis='y');ax.set_axisbelow(True)
    save(fig,output,'tolerance-calibration')
    gallery=read(root/'gallery.json')
    for category,g in gallery.items():
        if not g:continue
        fig,axes=plt.subplots(1,3,figsize=(14,3.7),layout='constrained');entry=dict(scene=g['scene'],goal=g['goal'],edit_m=g['row']['edit_m']);scene_axes(axes[0],entry)
        for p,color,ls,label in [(g['original']['states'],GREY,'-','Original'),(g['verified']['states'],GREEN,'-','Verified center'),(g['predicted'],BLUE,'--','Analytic prediction')]:
            p=np.array(p);axes[0].plot(p[:,0],p[:,1],color=color,ls=ls,label=label)
        axes[0].legend(fontsize=9);axes[0].set_title('(a) A nudge with room to spare' if category=='success' else '(a) A tolerance prediction can fail')
        grid=np.array(g['grid_m'])*1000;actual=np.array(g['actual_margins'],dtype=float);pred=np.array(g['predicted_margins'],dtype=float)
        for ax in axes[1:]:
            ax.plot(grid,actual,color=GREEN,label='Simulator');ax.plot(grid,pred,color=BLUE,ls='--',label='Prediction');ax.axhline(0,color=GREY,lw=1)
            for c in g['components']:ax.axvspan(c['low_m']*1000,c['high_m']*1000,color=GREEN,alpha=.1)
            ax.axvline(g['row']['edit_m']*1000,color=RUST,ls=':',label='Tolerant choice');ax.axvspan((g['row']['edit_m']-.005)*1000,(g['row']['edit_m']+.005)*1000,color=RUST,alpha=.1)
            if g['nominal_edit_m'] is not None:ax.axvline(g['nominal_edit_m']*1000,color=GREY,ls=':',label='Nominal choice')
            ax.set(xlabel='Bumper displacement (mm)',ylabel='Goal margin (m)');ax.grid(True)
        axes[1].set_title('(b) Measured successful positions');axes[1].legend(fontsize=8,ncol=2)
        center=g['row']['edit_m']*1000;nominal=(g['nominal_edit_m'] if g['nominal_edit_m'] is not None else g['row']['edit_m'])*1000;low=min(center,nominal)-10;high=max(center,nominal)+10
        axes[2].set_xlim(low,high);within=(grid>=low)&(grid<=high);values=np.r_[actual[within],pred[within],0.];vmin,vmax=np.nanmin(values),np.nanmax(values);pad=max(.01,(vmax-vmin)*.08);axes[2].set_ylim(vmin-pad,vmax+pad)
        for source,color,marker in [(g['nominal_checks'],GREY,'o'),(g['row']['checks'],RUST,'s')]:
            reference=g['nominal_edit_m'] if marker=='o' else g['row']['edit_m']
            points=[r for r in source if r['valid'] and r['margin'] is not None]
            axes[2].scatter([(reference+r['offset_m'])*1000 for r in points],[r['margin'] for r in points],color=color,marker=marker,s=22,zorder=5)
        axes[2].set_title('(c) Five measured execution errors');save(fig,output,f'tolerance-{category}')
    calls={stage:{k:sum(read(p)['calls'][k] for p in (root/f'{stage}-parents').glob('*.json')) for k in ['original','center','offline_perturbation','oracle']} for stage in ['calibration','evaluation']}
    return dict(protocol=config,models=summaries,paired_edits=paired_edits,paired_contrasts=contrasts,calls=calls,calibration=read(root/'calibration.json'),gallery_available={k:bool(v) for k,v in gallery.items()})


def demonstration_figures(output):
    record=read('artifacts/tolerance-study/demonstration.json');g=record['examples']['success']
    boundary=next(b for b in g['boundaries'] if b['kind']=='contact_association')
    fig,axes=plt.subplots(1,2,figsize=(12,3.8),layout='constrained')
    labels=[]
    for i,(hz,which) in enumerate([(1920,'outside'),(1920,'inside'),(3840,'outside'),(3840,'inside')]):
        events=boundary['events'][str(hz)][which];begins=[e['time_s'] for e in events if e['phase']=='begin' and e['object']=='bumper']
        axes[0].scatter(begins,[i]*len(begins),marker='|',s=65,color=GREEN if which=='inside' else RUST)
        labels.append(f'{hz} Hz · {which} ({len(begins)})')
    axes[0].set(yticks=range(4),yticklabels=labels,xlabel='Time after release (s)',title='(a) Repeated bumper-contact onsets')
    axes[0].grid(True,axis='x')
    coarse=read('artifacts/tolerance-study/gallery.json')['success'];grid=np.array(g['grid_m'])*1000
    for margins,color,ls,label in [(coarse['actual_margins'],GREEN,'-','1920 Hz'),(g['fine_margins'],BLUE,'--','3840 Hz')]:
        axes[1].plot(grid,margins,color=color,ls=ls,label=label)
    u=g['edit_m']*1000;c=g['common_component']
    axes[1].axvspan(c['low_m']*1000,c['high_m']*1000,color=GREEN,alpha=.1,label='Pass at both rates, sampled grid')
    axes[1].axvspan(u-5,u+5,color=RUST,alpha=.18,label='Requested ±5 mm');axes[1].axvline(u,color=RUST,ls=':')
    axes[1].axhline(0,color=GREY,lw=1);axes[1].set(xlim=(45,125),xlabel='Bumper displacement (mm)',ylabel='Goal margin (m)',title='(b) Stable goal labels, sensitive events');axes[1].grid(True);axes[1].legend(fontsize=9)
    save(fig,output,'tolerance-events');return record


def build(output,development_only=False):
    output.mkdir(parents=True,exist_ok=True);(output/'report-kind.json').write_text(json.dumps({'kind':'contact-followup-v1'}));style();summary=dict(training=training_figures(output),episodes=episode_figures(output),integrator=integrator_figures(output))
    if not development_only:
        summary['tolerance']=tolerance_figures(output)
        summary['demonstration']=demonstration_figures(output)
    summary['uncertainty']='Three fixed training seeds; paired parent bootstrap for tolerance effects, 2000 draws. Bands in episode figures span seeds, not confidence intervals. No continuous execution or population-risk guarantee.'
    (output/('development.json' if development_only else 'summary.json')).write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/contact-followup'));p.add_argument('--development-only',action='store_true');a=p.parse_args();build(a.output,a.development_only)
