def calculate_precision(tp: int, fp: int) -> float:
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def calculate_recall(tp: int, fn: int) -> float:
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)
