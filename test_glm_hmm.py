import pytest
from sklearn.utils.estimator_checks import check_estimator
from _glm_hmm import GLMHMMEstimator

@pytest.mark.parametrize(
    "Estimator", [GLMHMMEstimator]
)

def test_all_estimators(Estimator):
    return check_estimator(Estimator)