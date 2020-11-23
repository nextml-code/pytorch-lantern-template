import wildfire


def gradient_metrics():
    return dict(
        loss=wildfire.MapMetric(lambda examples, predictions, loss: loss),
    )


def evaluate_metrics():
    return dict(
        loss=wildfire.MapMetric(lambda examples, predictions, loss: loss),
    )
