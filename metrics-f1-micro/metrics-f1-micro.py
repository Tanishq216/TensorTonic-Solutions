def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    if len(y_true) == 0:
        return 0.0
        
    tp = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            tp += 1
    return float(tp / len(y_true))
    pass