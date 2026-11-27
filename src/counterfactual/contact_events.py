"""Callback episode semantics, kept separate from interval-occupancy classes."""
import numpy as np

EVENT_NAMES = ['occupancy','begin','separate','active_end','impact','support']


def event_labels(trajectory, ball_id='ball'):
    n,j=trajectory.contacts.shape
    labels=np.zeros((n,5,6),np.float32);labels[:,:j,0]=trajectory.contacts
    start=np.zeros((n,5),np.float32);times=np.zeros((n,5,2),np.float32)
    time_mask=np.zeros_like(times);counts=np.zeros((n,5,2),np.int32)
    grouped=[[] for _ in range(n)];ordered=[]
    for sequence,event in enumerate(trajectory.events):
        if event['phase'] not in ['begin','separate']:continue
        t=min(n-1,max(0,int(np.ceil(event['time']*60-1e-7))-1))
        obj=next(x for x in event['objects'] if x!=ball_id);edge=trajectory.object_ids.index(obj)
        kind=int(event['phase']=='separate');fraction=event['time']*60-t
        grouped[t].append((event['time'],sequence,edge,kind))
        counts[t,edge,kind]+=1;labels[t,edge,1+kind]=1
        if not time_mask[t,edge,kind]:times[t,edge,kind]=fraction;time_mask[t,edge,kind]=1
        ordered.append(dict(interval=t,fraction=float(fraction),object=obj,phase=event['phase'],time_s=event['time']))
    active=np.zeros(5,np.float32)
    for t,events in enumerate(grouped):
        start[t]=active
        for _,_,edge,kind in sorted(events):active[edge]=1-kind
        labels[t,:,3]=active
    for response in trajectory.responses:
        t=min(n-1,max(0,int(np.ceil(response['time']*60-1e-7))-1));edge=trajectory.object_ids.index(response['object'])
        normal=-np.asarray(response['normal_ball_to_static']);vn=np.asarray(response['before'][2:4])@normal
        impulse=np.asarray(response['impulse_on_ball'])@normal
        if impulse>1e-8:labels[t,edge,4 if vn<-.2 else 5]=1
    return dict(event_labels=labels,active_start=start,event_times=times,event_time_mask=time_mask,
                event_counts=counts,ordered_events=ordered)
