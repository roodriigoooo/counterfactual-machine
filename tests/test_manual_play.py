"""Player-chosen moves must retain the same edit and simulation boundaries."""
from pathlib import Path
from fastapi.testclient import TestClient
from counterfactual import api
from counterfactual.schemas import GoalSpec, Region
from counterfactual.scenes import generate_scene


def test_manual_move_is_bounded_and_validation_does_not_simulate(tmp_path, monkeypatch):
    seen = []
    simulate = api.simulate
    def counted(scene, edit=0):
        seen.append((scene.content_hash(), edit))
        return simulate(scene, edit)
    monkeypatch.setattr(api, 'simulate', counted)
    scene = generate_scene(0)
    task = dict(scene=scene.model_dump(), goal=GoalSpec(region=Region(radius=99)).model_dump(), edit_m=.01)
    client = TestClient(api.create_app(tmp_path/'absent.pt', tmp_path))
    assert client.post('/api/validate', json=task).json() == dict(valid=True, simulator_calls=0)
    assert not seen
    response = client.post('/api/try', json=task)
    assert response.status_code == 200
    assert response.json()['simulator_calls'] == 1
    assert len(response.json()['trajectory']['states']) == 121
    assert seen == [(scene.content_hash(), .01)]
    for invalid in [{**task, 'edit_m':scene.ball.radius+.001},
                    {**task, 'goal':{**task['goal'], 'ball_id':'missing'}},
                    {**task, 'edit_m':'not a number'}]:
        assert client.post('/api/try', json=invalid).status_code == 422
    assert len(seen) == 1
    # A manual success must never become a known ORIGINAL success for the planner.
    assert client.post('/api/plan', json={k:task[k] for k in ['scene','goal']}).status_code == 503


def test_import_and_manual_overlap_fail_before_simulation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid scene must not simulate')
    monkeypatch.setattr(api,'simulate',forbidden)
    scene = generate_scene(0).model_dump()
    scene['bumper']['position'] = scene['ball']['position']
    task = dict(scene=scene, goal=GoalSpec().model_dump())
    client = TestClient(api.create_app(Path('missing.pt'), tmp_path))
    for endpoint in ['validate','try']:
        response = client.post('/api/'+endpoint, json=task)
        assert response.status_code == 422 and 'overlap' in response.json()['detail']


def test_named_research_gallery_exposes_only_playable_tasks(tmp_path):
    import json
    scene=generate_scene(0).model_dump();goal=GoalSpec().model_dump()
    (tmp_path/'gallery.json').write_text(json.dumps({'contact_transition':{'scene':scene},'success':{'scene':scene,'goal':goal},'failure':{'scene':scene,'goal':goal}}))
    client=TestClient(api.create_app(tmp_path/'missing.pt',tmp_path))
    response=client.get('/api/presets')
    assert response.status_code==200 and len(response.json())==6
    assert all('goal' in task and 'scene' in task for task in response.json())
    assert client.get('/research-figures/model.pt').status_code==404


def test_contact_notebook_serves_saved_measurements_without_running_physics(tmp_path, monkeypatch):
    import json
    monkeypatch.chdir(tmp_path)
    root=tmp_path/'artifacts/contact-followup';root.mkdir(parents=True)
    example=dict(scene=generate_scene(46000).model_dump(mode='json'),goal=GoalSpec().model_dump(mode='json'),edit_m=.061875)
    (root/'summary.json').write_text(json.dumps(dict(tolerance={'cohort':'fresh'},demonstration={'examples':{'success':example}})))
    (root/'tolerance-events.svg').write_text('<svg/>')
    (root/'tolerance-events.png').write_bytes(b'figure')
    (root/'model.pt').write_bytes(b'private checkpoint')
    def forbidden(*args, **kwargs):
        raise AssertionError('Reading measured research must not simulate')
    monkeypatch.setattr(api,'simulate',forbidden)
    client=TestClient(api.create_app(tmp_path/'absent.pt',tmp_path/'absent'))
    record=client.get('/api/contact-research').json()
    assert record['example']==example and record['figures']==['tolerance-events']
    assert client.get('/contact-figures/tolerance-events.svg').status_code==200
    for path in ['model.pt','summary.json','tolerance-missing.svg','%2e%2e%2fmodel.pt']:
        assert client.get('/contact-figures/'+path).status_code==404
