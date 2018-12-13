def acc(preds, targs, th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds == targs).float().mean()


def f1(preds, targs, th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return preds * 2 + targs


def macrof1(f1_score):
    TP = (f1_score == 3).float() * 2
    other = TP + (f1_score == 2).float() + (f1_score == 1).float()
    return (TP / other).mean().cpu().numpy()
