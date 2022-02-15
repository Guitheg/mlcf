def L2(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean().item()
