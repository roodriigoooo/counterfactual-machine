from types import SimpleNamespace

from counterfactual.competition_scenes import begins, opportunities, propose, same_encounter, witness
from counterfactual.simulator import simulate


def test_callback_at_boundary_belongs_to_preceding_interval():
    tr = SimpleNamespace(events=[dict(phase="begin",time=13/60),dict(phase="begin",time=14/60)])
    assert begins(tr,13) == [tr.events[1]]


def test_two_rates_recover_a_closer_surface_only_when_the_winner_is_removed():
    scene = propose(62023)
    audits = []
    for hz in [1920,3840]:
        trajectory = simulate(scene,physics_hz=hz)
        race = witness(scene,trajectory,opportunities(scene,trajectory))
        assert race["nearest"] == race["omitted"] == "bumper"
        assert race["winner"] == "ramp"
        assert race["begin_s"] < race["isolated_begin_s"]
        audits.append(race)
    assert same_encounter(*audits)
    assert not same_encounter(audits[0],dict(audits[1],tick=audits[1]["tick"]+1))


def test_two_actual_contacts_do_not_qualify_as_preemption():
    scene = propose(62023)
    tr = simulate(scene)
    race = witness(scene,tr,opportunities(scene,tr))
    tick = race["tick"]
    tr.contacts[tick,tr.object_ids.index(race["omitted"])] = 1
    assert witness(scene,tr,[race]) is None


def test_replica_only_permits_reordering_concurrent_callbacks_on_distinct_objects():
    import copy
    import pytest
    from counterfactual.competition_audit import compare_manifests
    events=[dict(time_s=.5,object="bumper",phase="begin"),
            dict(time_s=.5,object="bumper",phase="separate"),
            dict(time_s=.5,object="ramp",phase="begin")]
    manifest={'records':[{'branches':[{'callbacks':events}]}]}
    def reordered(indices):
        replica=copy.deepcopy(manifest)
        replica['records'][0]['branches'][0]['callbacks']=[events[i] for i in indices]
        return replica
    assert compare_manifests(manifest,reordered([2,0,1]))==1
    with pytest.raises(ValueError):
        compare_manifests(manifest,reordered([1,2,0]))
