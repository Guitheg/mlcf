from torch import Tensor, tensor


"""
Shape of an y : (BatchSize, Ndim)
"""


def _et(y_true: Tensor, y_pred: Tensor):
    return (y_pred - y_true).abs()


def _pt(y_true: Tensor, y_pred: Tensor):
    return 100 * (_et(y_true, y_pred).abs() / y_true)


def _qj(y_true: Tensor, y_pred: Tensor, m: int = 1):
    ej = _et(y_true, y_pred)
    return ej / (_et(y_true[m:], y_true.roll(m, 0)[m:]).sum() / (y_true.size(0) - m))


def _eq(y_true: Tensor, y_pred: Tensor, tol: float = 0.01):
    std = tol * y_true.std()
    return (y_pred >= (y_true - std)) & (y_pred <= (y_true + std))


def _sup(y_true: Tensor, y_pred: Tensor, tol: float = 0.01):
    std = tol * y_true.std()
    return (y_pred > (y_true + std))


def _inf(y_true: Tensor, y_pred: Tensor, tol: float = 0.01):
    std = tol * y_true.std()
    return (y_pred < (y_true - std))


def mae(y_true: Tensor, y_pred: Tensor):
    """Mean Average Error"""
    return _et(y_true, y_pred).mean().item()


def rmse(y_true: Tensor, y_pred: Tensor):
    """Root Mean Square Error"""
    return _et(y_true, y_pred).pow(2).mean().sqrt().item()


def mape(y_true: Tensor, y_pred: Tensor):
    """Mean Absolute Percentage Error"""
    return _pt(y_true, y_pred).abs().mean().item()


def smape(y_true: Tensor, y_pred: Tensor):
    """Symmetric Mean Absolute Percentage Error"""
    return (200*_et(y_true, y_pred) / (y_true + y_pred)).mean().item()


def mase(y_true: Tensor, y_pred: Tensor, m: int = 1):
    """Mean Absolute Scaled Error"""
    return _qj(y_true, y_pred, m).mean().item()


def rrse(y_true: Tensor, y_pred: Tensor):
    """Root Relative Squarred Error"""
    return (
        _et(y_true, y_pred).pow(2).sum() /
        _et(y_true, y_true.mean()).pow(2).sum()
    ).sqrt().item()


def rae(y_true: Tensor, y_pred: Tensor):
    return (
        _et(y_true, y_pred).pow(2).sum() /
        _et(y_true, y_true.mean()).pow(2).sum()
    ).item()


def accu_neighborhood(y_true: Tensor, y_pred: Tensor):
    return (100 * _eq(y_true, y_pred).sum().item() / tensor(y_true.size()).prod()).item()


def oratio(y_true: Tensor, y_pred: Tensor):
    """Overestimation ratio"""
    return (100 * _sup(y_true, y_pred).sum().item() / tensor(y_true.size()).prod()).item()


def uratio(y_true: Tensor, y_pred: Tensor):
    """ Underestimation ratio"""
    return (100 * _inf(y_true, y_pred).sum().item() / tensor(y_true.size()).prod()).item()


def stde(y_true: Tensor, y_pred: Tensor):
    """Standard Deviation Error"""
    return _et(y_true, y_pred).abs().std().item()


def nrmse(y_true: Tensor, y_pred: Tensor):
    """Normalized Root Mean Square Error"""
    return (rmse(y_true, y_pred) / y_true.mean()).item()
