def calculate_f05(precision: float, recall: float) -> float:
    """Calculates F0.5 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)
