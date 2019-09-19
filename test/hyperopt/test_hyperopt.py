import hashlib
import pickle

from stplfcnn.hyperopt import HyperparameterOptimizer
from stplfcnn.utils import Reference


def test_calculate_params_hash():
    hash1 = HyperparameterOptimizer.calculate_params_hash(dict(
        type="STLQFCNNEstimator",
        time_horizon=Reference("time_horizon"),
        quantile_levels=Reference("quantile_levels"),
        history_shape=[7, 24],
    ))
    hash2 = hashlib.md5(pickle.dumps([
        ("history_shape", [7, 24]),
        ("quantile_levels", Reference("quantile_levels")),
        ("time_horizon", Reference("time_horizon")),
        ("type", "STLQFCNNEstimator"),
    ], pickle.HIGHEST_PROTOCOL)).hexdigest()
    assert isinstance(hash1, str)
    assert hash1 == hash2
