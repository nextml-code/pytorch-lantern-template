import numpy as np
import wildfire


def train_metrics():
    return dict(
        loss=wildfire.MapMetric(lambda examples, predictions, loss: loss),
    )


def progress_metrics():
    return dict(
        loss=wildfire.MapMetric(lambda examples, predictions, loss: loss),
    )


def evaluate_metrics():
    return dict(
        loss=wildfire.MapMetric(lambda examples, predictions, loss: loss),
    )
