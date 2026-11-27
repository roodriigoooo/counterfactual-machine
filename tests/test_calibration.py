from counterfactual.calibration import THRESHOLDS,choose_threshold


def rows(selected=20,failures=0):
    return [dict(parent_id=f'p{i//2}',kind='reach' if i%2 else 'avoid',selected=i<selected,success=failures<=i<selected) for i in range(30)]


def test_threshold_selected_after_search_coverage_and_failure_counts():
    cases={str(t):rows(selected=10,failures=0) for t in THRESHOLDS}
    cases['0.0']=rows(selected=30,failures=8)
    cases['0.01']=rows(selected=25,failures=3)
    cases['0.03']=rows(selected=20,failures=1)
    assert choose_threshold(cases)==.03


def test_no_qualifying_threshold_is_abstention_not_implicit_zero():
    cases={str(t):rows(selected=9,failures=0) for t in THRESHOLDS}
    assert choose_threshold(cases) is None
