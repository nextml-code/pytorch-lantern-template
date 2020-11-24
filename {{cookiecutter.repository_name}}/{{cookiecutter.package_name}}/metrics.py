import lantern


def gradient_metrics():
    return dict(
        loss=lantern.MapMetric(lambda examples, predictions, loss: loss),
    )


def evaluate_metrics():
    return dict(
        loss=lantern.MapMetric(lambda examples, predictions, loss: loss),
    )
