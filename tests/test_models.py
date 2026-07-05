from app.models import rf
from app.models import xgboost_model
from app.models import lightgbm_model


def test_rf_build_estimator_applies_class_weight():
    estimator = rf.build_estimator(class_weight={0: 0.5, 1: 5.0}, random_state=42)
    assert estimator.class_weight == {0: 0.5, 1: 5.0}
    assert estimator.random_state == 42


def test_rf_param_distributions_is_nonempty_dict():
    params = rf.param_distributions()
    assert isinstance(params, dict)
    assert "n_estimators" in params
    assert "max_depth" in params


def test_xgboost_build_estimator_applies_scale_pos_weight():
    estimator = xgboost_model.build_estimator(scale_pos_weight=9.0, random_state=42)
    assert estimator.get_params()["scale_pos_weight"] == 9.0
    assert estimator.get_params()["random_state"] == 42


def test_xgboost_param_distributions_is_nonempty_dict():
    params = xgboost_model.param_distributions()
    assert isinstance(params, dict)
    assert "max_depth" in params
    assert "learning_rate" in params


def test_lightgbm_build_estimator_applies_class_weight():
    estimator = lightgbm_model.build_estimator(class_weight={0: 0.5, 1: 5.0}, random_state=42)
    assert estimator.get_params()["class_weight"] == {0: 0.5, 1: 5.0}
    assert estimator.get_params()["random_state"] == 42


def test_lightgbm_param_distributions_is_nonempty_dict():
    params = lightgbm_model.param_distributions()
    assert isinstance(params, dict)
    assert "num_leaves" in params
    assert "learning_rate" in params
