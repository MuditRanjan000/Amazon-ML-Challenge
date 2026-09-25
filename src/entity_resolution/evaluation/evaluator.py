class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, ground_truth, predictions):
        """
        Evaluate predictions against ground truth using the official competition F0.5 macro-average metric.
        Must handle singleton evaluation correctly.
        """
        raise NotImplementedError
