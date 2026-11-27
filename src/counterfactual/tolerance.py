"""Finite-grid execution tolerance, distinct from goal clearance and uncertainty."""
import numpy as np
from .planning import select_index


def neighborhoods(grid, epsilon, direction='symmetric'):
    grid=np.asarray(grid,float)
    if epsilon<0 or not np.isfinite(epsilon):raise ValueError('epsilon must be finite and nonnegative')
    if len(grid)<3 or not np.allclose(np.diff(grid),grid[1]-grid[0],rtol=1e-8,atol=1e-12):raise ValueError('uniform grid required')
    if direction not in ['symmetric','overshoot']:raise ValueError('unknown tolerance direction')
    steps=int(np.ceil(epsilon/(grid[1]-grid[0])-1e-10))
    result=[]
    for i,u in enumerate(grid):
        # At zero an overshoot direction is undefined: use both directions.
        low=i-steps if direction=='symmetric' or u<=0 else i
        high=i+steps if direction=='symmetric' or u>=0 else i
        result.append(None if low<0 or high>=len(grid) else np.arange(low,high+1))
    return result,steps*(grid[1]-grid[0])


def select_with_tolerance(grid,margins,passed,valid,epsilon=0.,gamma=.03,direction='symmetric'):
    if gamma is None:return dict(index=None,effective_epsilon_m=None,worst_margins=None)
    if not np.isfinite(gamma) or gamma<0:raise ValueError('gamma must be finite and nonnegative')
    grid=np.asarray(grid,float);margins=np.asarray(margins,float);passed=np.asarray(passed,bool);valid=np.asarray(valid,bool)
    windows,effective=neighborhoods(grid,epsilon,direction);worst=np.full(len(grid),np.nan);accepted=np.zeros(len(grid),bool)
    for i,window in enumerate(windows):
        if window is None or not valid[window].all() or not np.isfinite(margins[window]).all():continue
        worst[i]=margins[window].min()
        accepted[i]=passed[window].all() and worst[i]>=gamma
    return dict(index=select_index(grid,worst,accepted),effective_epsilon_m=float(effective),worst_margins=worst)


def successful_components(grid,passed):
    grid=np.asarray(grid,float);passed=np.asarray(passed,bool);result=[];start=None
    for i,success in enumerate(np.r_[passed,False]):
        if success and start is None:start=i
        elif not success and start is not None:
            result.append(dict(first=start,last=i-1,low_m=float(grid[start]),high_m=float(grid[i-1])));start=None
    return result
