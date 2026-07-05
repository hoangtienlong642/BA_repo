from app.models import rf


def test_rf_build_estimator_applies_class_weight():
    estimator = rf.build_estimator(class_weight={0: 0.5, 1: 5.0}, random_state=42)
    assert estimator.class_weight == {0: 0.5, 1: 5.0}
    assert estimator.random_state == 42


def test_rf_param_distributions_is_nonempty_dict():
    params = rf.param_distributions()
    assert isinstance(params, dict)
    assert "n_estimators" in params
    assert "max_depth" in params
