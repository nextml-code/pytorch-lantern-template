import numpy as np
import lantern


def train_metrics():
    return dict(
        loss=lantern.ReduceMetric(lambda state, loss: loss.item()),
        accuracy=lantern.ReduceMetric(
            reduce_fn=lambda rolling_mean, examples, predictions: (
                0.9 * rolling_mean
                + 0.1
                * np.mean(
                    [
                        example.class_name == prediction.class_name
                        for example, prediction in zip(examples, predictions)
                    ]
                )
            ),
            initial_state=0,
        ),
    )


def evaluate_metrics():
    return dict(
        loss=lantern.MapMetric(lambda loss: loss.item()),
        accuracy=lantern.MapMetric(
            map_fn=lambda examples, predictions: [
                example.class_name == prediction.class_name
                for example, prediction in zip(examples, predictions)
            ],
            compute_fn=lambda results: np.mean(np.concatenate(results)),
        ),
    )
