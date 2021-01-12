import lantern


def gradient_metrics():
    return dict(
        loss=lantern.ReduceMetric(lambda state, examples, predictions, loss: loss),
    )


def evaluate_metrics():
    return dict(
        loss=lantern.MapMetric(lambda examples, predictions, loss: loss),
    )
