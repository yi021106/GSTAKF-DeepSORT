from .gstakf import GaussianSTAKFFilter
from .nsa import NSAOnlyFilter
from .standard import StandardKalmanFilter
from .stf import STFOnlyFilter


def create_filter(filter_type="standard", **parameters):
    if filter_type == "standard":
        return StandardKalmanFilter()
    if filter_type == "nsa":
        return NSAOnlyFilter()
    if filter_type == "stf":
        return STFOnlyFilter(
            rho=parameters.get("rho", 0.82),
            beta=parameters.get("beta", 1.0),
        )
    if filter_type == "gstakf":
        return GaussianSTAKFFilter(
            rho=parameters.get("rho", 0.82),
            beta=parameters.get("beta", 1.0),
            lambda_center=parameters.get("lambda_center", 0.0),
            conf_center=parameters.get("conf_center", 0.5),
            sigma_lambda=parameters.get("sigma_lambda", 0.5),
            sigma_conf=parameters.get("sigma_conf", 0.3),
            sigmoid_k=parameters.get("sigmoid_k", 8.0),
            sigmoid_x0=parameters.get("sigmoid_x0", 0.75),
        )
    raise ValueError(filter_type)
